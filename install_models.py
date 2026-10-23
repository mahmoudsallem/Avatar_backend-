"""
install_models.py
------------------
Uses an S3 bucket as a cache for the ML models this service downloads, so a new EC2 instance
doesn't have to re-pull everything from Hugging Face / DeepFace (and re-accept the gated FLUX
license) on every boot.

Mirrors three local directories <-> s3://<S3_MODEL_BUCKET>/<S3_MODEL_CACHE_PREFIX>. This is a
separate bucket from the one the photo pipeline (app/s3_watcher.py) uses:

  huggingface_hub/   <-> ~/.cache/huggingface/hub (or $HF_HOME/hub)
                         Only the specific repos app/config.py currently points at are synced -
                         NUNCHAKU_MODEL or FLUX_MODEL (whichever FLUX_BACKEND actually uses),
                         LORA, and CLIP_MODEL. Anything else sitting in this cache (old A/B
                         experiments, a model someone tried out, an interrupted download) is
                         left alone - the HF cache is shared/cumulative and routinely ends up
                         holding things this app doesn't use.
  deepface_weights/  <-> ~/.deepface/weights (or $DEEPFACE_HOME/weights)
                         RetinaFace detector, ArcFace recognition, gender classifier - these
                         are auto-downloaded by DeepFace on first use (app/deepface_worker.py).
  local_clip/        <-> /Models/clip-vit-base-patch32 (CLIP_DIR in app/config.py), only if
                         that directory actually exists on this box.

Usage:
    # On the FIRST instance: run the app once so the caches above are populated, then push
    # them up to S3 to seed the bucket:
    python install_models.py push

    # On every NEW instance: pull everything down before starting the app, so nothing needs
    # to be re-downloaded from Hugging Face / DeepFace:
    python install_models.py pull

    # One-shot setup for a new box: pull from S3, download anything still missing straight from
    # Hugging Face (e.g. the gated FLUX.2-klein-9B base model - needs HF_TOKEN or `hf auth login`
    # from an account that accepted the license), then push it back to S3 so the next box only
    # needs `pull`:
    python install_models.py install

    # Only sync one of the three sets, or preview without transferring anything:
    python install_models.py pull --only huggingface_hub
    python install_models.py push --dry-run

AWS credentials come from the normal boto3 chain (env vars, ~/.aws/credentials, or an EC2
instance role) - same as app/s3_watcher.py. Bucket/region/prefix come from app/config.py
(S3_MODEL_BUCKET, S3_REGION, S3_MODEL_CACHE_PREFIX), so they follow the same .env as the rest
of the service.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

HF_CACHE_DIR = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
DEEPFACE_WEIGHTS_DIR = Path(os.environ.get("DEEPFACE_HOME", "~/.deepface")).expanduser() / "weights"
MANIFEST_NAME = "_symlinks_manifest.json"
# HF's own housekeeping (lock files / "checked and it doesn't exist" markers) - not model data.
_SKIP_PARTS = {".locks", ".no_exist"}
# A download that got interrupted - never a complete, loadable weight file regardless of model.
_SKIP_SUFFIX = ".incomplete"

# name -> (local dir, is this an HF hub cache - i.e. blobs/ + symlinked snapshots/ - or a flat dir?)
MODEL_SETS: List[Tuple[str, Path, bool]] = [
    ("huggingface_hub", HF_CACHE_DIR, True),
    ("deepface_weights", DEEPFACE_WEIGHTS_DIR, False),
    ("local_clip", settings.CLIP_DIR, False),
]


def _configured_hf_repos() -> List[str]:
    """HF repo IDs this app actually loads right now, per the current app/config.py settings -
    not everything that happens to be sitting in the cache. A "repo" that's actually a local
    directory path (e.g. LORA or CLIP_MODEL pointed at a folder instead of a HF repo id) has
    nothing to sync here - that's not stored in the HF hub cache at all."""
    repos: List[str] = []

    def _add(repo_id: str) -> None:
        if repo_id and not Path(repo_id).is_dir() and repo_id not in repos:
            repos.append(repo_id)

    _add(settings.NUNCHAKU_MODEL if settings.FLUX_BACKEND == "nunchaku" else settings.FLUX_MODEL)
    _add(settings.LORA)
    _add(settings.CLIP_MODEL)
    return repos


def _configured_hf_repo_dirs() -> List[str]:
    """Each configured repo id's cache directory name, e.g. "black-forest-labs/FLUX.2-klein-9B"
    -> "models--black-forest-labs--FLUX.2-klein-9B" (HF's own naming convention)."""
    return [f"models--{repo.replace('/', '--')}" for repo in _configured_hf_repos()]


def _hf_token() -> Optional[str]:
    """HF_TOKEN / HUGGING_FACE_HUB_TOKEN from the environment, else None (huggingface_hub then falls
    back to the token saved by `hf auth login`). Never read from .env - that file is committed to git."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def hf_download_missing(dry_run: bool) -> bool:
    """Downloads every configured HF repo (see _configured_hf_repos) that is not fully in the local
    HF cache yet, straight from Hugging Face. snapshot_download is idempotent: complete files are
    skipped. The LoRA repo only fetches LORA_FILE. Returns False if a repo could not be fetched."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    ok = True
    for repo in _configured_hf_repos():
        patterns = [settings.LORA_FILE] if repo == settings.LORA and settings.LORA_FILE else None
        if dry_run:
            print(f"[huggingface_hub] would ensure {repo} is downloaded" + (f" ({patterns[0]} only)" if patterns else ""))
            continue
        print(f"[huggingface_hub] ensuring {repo} is in {HF_CACHE_DIR} (skips files already there)...")
        try:
            snapshot_download(repo_id=repo, cache_dir=str(HF_CACHE_DIR), allow_patterns=patterns, token=_hf_token())
            print(f"[huggingface_hub] {repo}: OK")
        except GatedRepoError:
            ok = False
            print(f"  [FAIL] {repo} is gated. Accept the license at https://huggingface.co/{repo} with your account, then "
                  f"run `hf auth login` or `export HF_TOKEN=hf_...` (read token) and rerun.", file=sys.stderr)
        except RepositoryNotFoundError:
            ok = False
            print(f"  [FAIL] {repo}: not found, or the token has no access to it.", file=sys.stderr)
        except Exception as exc:
            ok = False
            print(f"  [FAIL] {repo}: {exc}", file=sys.stderr)
    print()
    return ok


def _client():
    return boto3.client("s3", region_name=settings.S3_REGION)


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _s3_objects(s3, bucket: str, prefix: str) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                sizes[obj["Key"]] = obj["Size"]
    return sizes


def _plain_files(root: Path) -> Iterable[Path]:
    """Every real (non-symlink) file under root, skipping HF's .locks/.no_exist housekeeping and
    any interrupted-download fragment."""
    for path in root.rglob("*"):
        if _SKIP_PARTS.intersection(path.parts) or path.name.endswith(_SKIP_SUFFIX):
            continue
        if path.is_file() and not path.is_symlink():
            yield path


def _hf_cache_contents(root: Path, repo_dirs: List[str]) -> Tuple[List[Path], List[Tuple[str, str]]]:
    """Splits the configured repos' cache dirs (only repo_dirs - e.g.
    "models--black-forest-labs--FLUX.2-klein-9B" - not the whole HF cache) into real files and
    symlinks. Snapshot dirs are almost entirely symlinks into blobs/ - without this split, every
    snapshot that links to a given blob would re-upload/re-download that blob's bytes again.
    These are multi-GB model weights, so that duplication is not an acceptable "simple"
    tradeoff. Relative paths are computed against root (the overall HF cache dir), so the S3
    layout stays e.g. "models--.../blobs/<hash>" regardless of which repos are configured."""
    root_resolved = root.resolve()
    real_files: List[Path] = []
    symlinks: List[Tuple[str, str]] = []
    for repo_dir in repo_dirs:
        repo_root = root / repo_dir
        if not repo_root.is_dir():
            print(f"  [WARN] configured model not in the cache, skipping: {repo_dir}", file=sys.stderr)
            continue
        real_files.extend(_plain_files(repo_root))
        for path in repo_root.rglob("*"):
            if _SKIP_PARTS.intersection(path.parts) or not path.is_symlink():
                continue
            target = path.resolve()
            if not target.exists():
                print(f"  [WARN] broken symlink, skipping: {path.relative_to(root)}", file=sys.stderr)
                continue
            try:
                target_rel = target.relative_to(root_resolved).as_posix()
            except ValueError:
                print(f"  [WARN] symlink points outside the cache, skipping: {path.relative_to(root)}", file=sys.stderr)
                continue
            symlinks.append((path.relative_to(root).as_posix(), target_rel))
    return real_files, symlinks


def _upload(s3, bucket: str, existing: Dict[str, int], items: List[Tuple[Path, str]], dry_run: bool) -> Tuple[int, int, int]:
    uploaded = skipped = 0
    uploaded_bytes = 0
    for path, key in items:
        size = path.stat().st_size
        if existing.get(key) == size:
            skipped += 1
            continue
        if dry_run:
            print(f"  [would upload] {key} ({_human(size)})")
        else:
            try:
                s3.upload_file(str(path), bucket, key)
                print(f"  [OK] {key} ({_human(size)})")
            except (BotoCoreError, ClientError) as exc:
                print(f"  [FAIL] {key}: {exc}", file=sys.stderr)
                continue
        uploaded += 1
        uploaded_bytes += size
    return uploaded, skipped, uploaded_bytes


def _download(s3, bucket: str, items: List[Tuple[str, int, Path]], dry_run: bool) -> Tuple[int, int, int]:
    downloaded = skipped = 0
    downloaded_bytes = 0
    for key, size, dest in items:
        if dest.exists() and dest.stat().st_size == size:
            skipped += 1
            continue
        if dry_run:
            print(f"  [would download] {key} ({_human(size)})")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(bucket, key, str(dest))
                print(f"  [OK] {key} ({_human(size)})")
            except (BotoCoreError, ClientError) as exc:
                print(f"  [FAIL] {key}: {exc}", file=sys.stderr)
                continue
        downloaded += 1
        downloaded_bytes += size
    return downloaded, skipped, downloaded_bytes


def push(s3, bucket: str, name: str, local_root: Path, s3_prefix: str, is_hf_cache: bool, dry_run: bool,
         extra_repos: Optional[List[str]] = None) -> None:
    if not local_root.is_dir():
        print(f"[{name}] skipping - {local_root} does not exist locally")
        return

    if is_hf_cache:
        # extra_repos backs up specific repos NOT in app/config.py (e.g. stray downloads you
        # want archived before running `huggingface-cli delete-cache` to reclaim disk) - on top
        # of, not instead of, the normally-configured ones.
        repo_dirs = _configured_hf_repo_dirs() + [f"models--{r.replace('/', '--')}" for r in (extra_repos or [])]
        real_files, symlinks = _hf_cache_contents(local_root, repo_dirs)
    else:
        real_files, symlinks = list(_plain_files(local_root)), []

    if not real_files:
        print(f"[{name}] skipping - {local_root} has no files")
        return

    existing = _s3_objects(s3, bucket, s3_prefix)
    items = [(p, s3_prefix + p.relative_to(local_root).as_posix()) for p in real_files]
    suffix = f" ({len(symlinks)} symlink(s) recorded)" if symlinks else ""
    print(f"[{name}] pushing {len(items)} file(s) from {local_root} -> s3://{bucket}/{s3_prefix}{suffix}")
    uploaded, skipped, uploaded_bytes = _upload(s3, bucket, existing, items, dry_run)

    if symlinks and not dry_run:
        # Merge into (not overwrite) any existing manifest - a differently-configured instance
        # may have pushed a different set of repos earlier, and this push only ever collects
        # symlinks for ITS OWN configured repos, not everything the bucket holds.
        manifest_key = s3_prefix + MANIFEST_NAME
        try:
            resp = s3.get_object(Bucket=bucket, Key=manifest_key)
            merged = dict(json.loads(resp["Body"].read()).get("symlinks", []))
        except (BotoCoreError, ClientError):
            merged = {}
        merged.update(symlinks)
        s3.put_object(Bucket=bucket, Key=manifest_key, Body=json.dumps({"symlinks": sorted(merged.items())}, indent=2).encode())

    print(f"[{name}] done: {uploaded} uploaded ({_human(uploaded_bytes)}), {skipped} already up to date")


def pull(s3, bucket: str, name: str, local_root: Path, s3_prefix: str, is_hf_cache: bool, dry_run: bool) -> None:
    objects = _s3_objects(s3, bucket, s3_prefix)
    manifest_key = s3_prefix + MANIFEST_NAME
    objects.pop(manifest_key, None)

    if is_hf_cache:
        # Only fetch repos THIS instance's config actually needs - the bucket may also hold
        # repos pushed by a differently-configured instance (e.g. a diffusers-backend box next
        # to a nunchaku-backend one).
        repo_dirs = _configured_hf_repo_dirs()
        allowed = tuple(s3_prefix + d + "/" for d in repo_dirs)
        objects = {k: v for k, v in objects.items() if k.startswith(allowed)}

    if not objects:
        print(f"[{name}] skipping - s3://{bucket}/{s3_prefix} is empty")
        return

    items = [(key, size, local_root / key[len(s3_prefix):]) for key, size in objects.items()]
    print(f"[{name}] pulling {len(items)} file(s) from s3://{bucket}/{s3_prefix} -> {local_root}")
    downloaded, skipped, downloaded_bytes = _download(s3, bucket, items, dry_run)
    print(f"[{name}] done: {downloaded} downloaded ({_human(downloaded_bytes)}), {skipped} already up to date")

    if is_hf_cache and not dry_run:
        try:
            resp = s3.get_object(Bucket=bucket, Key=manifest_key)
            manifest = json.loads(resp["Body"].read())
        except (BotoCoreError, ClientError):
            manifest = {"symlinks": []}
        repo_prefixes = tuple(d + "/" for d in _configured_hf_repo_dirs())
        recreated = 0
        for link_rel, target_rel in manifest.get("symlinks", []):
            if not link_rel.startswith(repo_prefixes):
                continue  # recorded by a differently-configured instance's push - not pulled here
            link_path = local_root / link_rel
            target_path = local_root / target_rel
            link_path.parent.mkdir(parents=True, exist_ok=True)
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            try:
                os.symlink(os.path.relpath(target_path, link_path.parent), link_path)
                recreated += 1
            except OSError as exc:
                print(f"  [WARN] could not recreate symlink {link_rel}: {exc}", file=sys.stderr)
        if recreated:
            print(f"[{name}] recreated {recreated} symlink(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["push", "pull", "install"], help="push: local caches -> S3. pull: S3 -> local caches. install: pull, then download whatever is still missing from Hugging Face, then push it to S3")
    parser.add_argument("--only", help="comma-separated subset of: " + ",".join(n for n, _, _ in MODEL_SETS))
    parser.add_argument("--dry-run", action="store_true", help="list what would transfer without moving data")
    parser.add_argument("--extra-repos", help="push only: comma-separated HF repo ids to back up IN ADDITION to the "
                                               "normally-configured ones (e.g. stray downloads you want archived "
                                               "before running `huggingface-cli delete-cache` to reclaim disk space)")
    args = parser.parse_args()

    wanted = set(args.only.split(",")) if args.only else None
    sets = [s for s in MODEL_SETS if wanted is None or s[0] in wanted]
    if not sets:
        parser.error(f"--only matched nothing; choices are {','.join(n for n, _, _ in MODEL_SETS)}")

    bucket = settings.S3_MODEL_BUCKET
    base_prefix = settings.S3_MODEL_CACHE_PREFIX
    s3 = _client()

    try:
        s3.head_bucket(Bucket=bucket)
    except (BotoCoreError, ClientError) as exc:
        print(f"[FAIL] cannot reach bucket s3://{bucket}: {exc}", file=sys.stderr)
        sys.exit(1)

    dry_suffix = "  (dry run)" if args.dry_run else ""
    print(f"Bucket: s3://{bucket}/{base_prefix}  |  action: {args.action}{dry_suffix}\n")

    extra_repos = [r.strip() for r in args.extra_repos.split(",")] if args.extra_repos else None
    if extra_repos and args.action not in ("push", "install"):
        parser.error("--extra-repos only applies to push")

    def run(action: str) -> None:
        for name, local_root, is_hf_cache in sets:
            s3_prefix = f"{base_prefix}{name}/"
            if action == "push":
                push(s3, bucket, name, local_root, s3_prefix, is_hf_cache, args.dry_run,
                     extra_repos=extra_repos if is_hf_cache else None)
            else:
                pull(s3, bucket, name, local_root, s3_prefix, is_hf_cache, args.dry_run)
            print()

    if args.action == "install":
        run("pull")
        hf_ok = True
        if wanted is None or "huggingface_hub" in wanted:
            hf_ok = hf_download_missing(args.dry_run)
        run("push")
        if not hf_ok:
            sys.exit(1)
    else:
        run(args.action)


if __name__ == "__main__":
    main()
