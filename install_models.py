"""
install_models.py
------------------
Uses an S3 bucket as a cache for the ML models this service downloads, so a new EC2 instance
doesn't have to re-pull everything from Hugging Face / DeepFace (and re-accept the gated FLUX
license) on every boot.

Mirrors three local directories <-> s3://<S3_MODEL_BUCKET>/<S3_MODEL_CACHE_PREFIX>. This is a
separate bucket from the one the photo pipeline (app/s3_watcher.py) uses:

  huggingface_hub/   <-> ~/.cache/huggingface/hub (or $HF_HOME/hub)
                         FLUX.2-klein-9B, the Nunchaku quantized FLUX.2-klein-9B (if
                         FLUX_BACKEND=nunchaku), the BFS LoRA, the CLIP model (if not using
                         the local /Models override below), and anything else
                         from_pretrained()/hf_hub_download() pulled - this is HF's own cache,
                         so whatever is in it is "every model actually used so far".
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
from typing import Dict, Iterable, List, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

HF_CACHE_DIR = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
DEEPFACE_WEIGHTS_DIR = Path(os.environ.get("DEEPFACE_HOME", "~/.deepface")).expanduser() / "weights"
MANIFEST_NAME = "_symlinks_manifest.json"
# HF's own housekeeping (lock files / "checked and it doesn't exist" markers) - not model data.
_SKIP_PARTS = {".locks", ".no_exist"}

# name -> (local dir, is this an HF hub cache - i.e. blobs/ + symlinked snapshots/ - or a flat dir?)
MODEL_SETS: List[Tuple[str, Path, bool]] = [
    ("huggingface_hub", HF_CACHE_DIR, True),
    ("deepface_weights", DEEPFACE_WEIGHTS_DIR, False),
    ("local_clip", settings.CLIP_DIR, False),
]


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
    """Every real (non-symlink) file under root, skipping HF's .locks/.no_exist housekeeping."""
    for path in root.rglob("*"):
        if _SKIP_PARTS.intersection(path.parts):
            continue
        if path.is_file() and not path.is_symlink():
            yield path


def _hf_cache_contents(root: Path) -> Tuple[List[Path], List[Tuple[str, str]]]:
    """Splits an HF hub cache dir into real files and symlinks. Snapshot dirs are almost
    entirely symlinks into blobs/ - without this split, every snapshot that links to a given
    blob would re-upload/re-download that blob's bytes again. These are multi-GB model
    weights, so that duplication is not an acceptable "simple" tradeoff."""
    root_resolved = root.resolve()
    real_files = list(_plain_files(root))
    symlinks: List[Tuple[str, str]] = []
    for path in root.rglob("*"):
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


def push(s3, bucket: str, name: str, local_root: Path, s3_prefix: str, is_hf_cache: bool, dry_run: bool) -> None:
    if not local_root.is_dir():
        print(f"[{name}] skipping - {local_root} does not exist locally")
        return

    if is_hf_cache:
        real_files, symlinks = _hf_cache_contents(local_root)
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
        manifest_key = s3_prefix + MANIFEST_NAME
        s3.put_object(Bucket=bucket, Key=manifest_key, Body=json.dumps({"symlinks": symlinks}, indent=2).encode())

    print(f"[{name}] done: {uploaded} uploaded ({_human(uploaded_bytes)}), {skipped} already up to date")


def pull(s3, bucket: str, name: str, local_root: Path, s3_prefix: str, is_hf_cache: bool, dry_run: bool) -> None:
    objects = _s3_objects(s3, bucket, s3_prefix)
    manifest_key = s3_prefix + MANIFEST_NAME
    objects.pop(manifest_key, None)
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
        for link_rel, target_rel in manifest.get("symlinks", []):
            link_path = local_root / link_rel
            target_path = local_root / target_rel
            link_path.parent.mkdir(parents=True, exist_ok=True)
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            try:
                os.symlink(os.path.relpath(target_path, link_path.parent), link_path)
            except OSError as exc:
                print(f"  [WARN] could not recreate symlink {link_rel}: {exc}", file=sys.stderr)
        if manifest.get("symlinks"):
            print(f"[{name}] recreated {len(manifest['symlinks'])} symlink(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["push", "pull"], help="push: local caches -> S3. pull: S3 -> local caches")
    parser.add_argument("--only", help="comma-separated subset of: " + ",".join(n for n, _, _ in MODEL_SETS))
    parser.add_argument("--dry-run", action="store_true", help="list what would transfer without moving data")
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

    for name, local_root, is_hf_cache in sets:
        s3_prefix = f"{base_prefix}{name}/"
        if args.action == "push":
            push(s3, bucket, name, local_root, s3_prefix, is_hf_cache, args.dry_run)
        else:
            pull(s3, bucket, name, local_root, s3_prefix, is_hf_cache, args.dry_run)
        print()


if __name__ == "__main__":
    main()
