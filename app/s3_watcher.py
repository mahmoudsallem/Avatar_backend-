"""
Polls s3://<S3_BUCKET>/<S3_USERS_PREFIX> for new user photos. Each new photo
is run through the same analysis + generation pipeline as the HTTP endpoints,
and the resulting avatar is uploaded to s3://<S3_BUCKET>/<S3_GENERATED_PREFIX>
under the same base filename (as a .jpg - the pipeline's output is always JPEG).

A photo counts as "already done" when a same-named file already exists in
S3_GENERATED_PREFIX, so the watcher is safe to restart without redoing work
or needing a separate database of what's processed.

Enable with S3_ENABLED=true in .env. AWS credentials come from the normal
boto3 chain (env vars, ~/.aws/credentials, or an EC2 instance role).
"""
import asyncio
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Set

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings
from app import pipeline

logger = logging.getLogger("saytara.s3_watcher")

_s3 = None
_task: Optional[asyncio.Task] = None
# Stems that failed this run - skipped until the process restarts, so a bad
# photo doesn't get retried (and burn a GPU slot) on every poll.
_failed_stems: Set[str] = set()


def _client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=settings.S3_REGION)
    return _s3


def _list_stems(s3, prefix: str) -> Set[str]:
    stems: Set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            stems.add(Path(key).stem)
    return stems


def _pending_keys(s3) -> List[str]:
    done_stems = _list_stems(s3, settings.S3_GENERATED_PREFIX)
    pending: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=settings.S3_USERS_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or obj["Size"] == 0:
                continue
            if Path(key).suffix.lower() not in pipeline.IMAGE_EXTENSIONS:
                continue
            stem = Path(key).stem
            if stem in done_stems or stem in _failed_stems:
                continue
            pending.append(key)
    return pending


async def _process_one(s3, key: str) -> None:
    stem = Path(key).stem
    settings.VAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = settings.VAL_TMP_DIR / f"s3_{uuid.uuid4().hex}{Path(key).suffix.lower()}"
    try:
        await asyncio.to_thread(s3.download_file, settings.S3_BUCKET, key, str(tmp_path))
        jpeg_bytes, metadata = await pipeline.process_image_file(tmp_path)

        out_key = f"{settings.S3_GENERATED_PREFIX}{stem}.jpg"
        await asyncio.to_thread(
            s3.put_object,
            Bucket=settings.S3_BUCKET,
            Key=out_key,
            Body=jpeg_bytes,
            ContentType="image/jpeg",
        )
        logger.info(
            "S3 watcher: s3://%s/%s -> s3://%s/%s (tries=%s, id_sim=%s)",
            settings.S3_BUCKET, key, settings.S3_BUCKET, out_key,
            metadata.get("tries"), metadata.get("id_sim"),
        )
    except pipeline.Rejected as r:
        logger.warning("S3 watcher: rejected s3://%s/%s: %s", settings.S3_BUCKET, key, r)
        _failed_stems.add(stem)
    except (BotoCoreError, ClientError):
        logger.exception("S3 watcher: S3 error handling s3://%s/%s", settings.S3_BUCKET, key)
    except Exception:
        logger.exception("S3 watcher: failed to process s3://%s/%s", settings.S3_BUCKET, key)
        _failed_stems.add(stem)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


async def _poll_once() -> None:
    s3 = _client()
    keys = await asyncio.to_thread(_pending_keys, s3)
    if not keys:
        return

    logger.info("S3 watcher: %d new user photo(s) to process", len(keys))
    sem = asyncio.Semaphore(settings.S3_MAX_CONCURRENT)

    async def _guarded(key: str) -> None:
        async with sem:
            await _process_one(s3, key)

    await asyncio.gather(*(_guarded(key) for key in keys))


async def watch_loop() -> None:
    logger.info(
        "S3 watcher started: s3://%s/%s -> s3://%s/%s (poll every %.0fs)",
        settings.S3_BUCKET, settings.S3_USERS_PREFIX,
        settings.S3_BUCKET, settings.S3_GENERATED_PREFIX,
        settings.S3_POLL_INTERVAL,
    )
    while True:
        try:
            await _poll_once()
        except (BotoCoreError, ClientError):
            logger.exception("S3 watcher: S3 error during poll")
        except Exception:
            logger.exception("S3 watcher: unexpected error during poll")
        await asyncio.sleep(settings.S3_POLL_INTERVAL)


def start() -> Optional[asyncio.Task]:
    """Start the background poll loop if S3_ENABLED; no-op otherwise."""
    global _task
    if not settings.S3_ENABLED:
        logger.info("S3 watcher disabled (S3_ENABLED=false)")
        return None
    _task = asyncio.create_task(watch_loop())
    return _task


def is_running() -> bool:
    return _task is not None and not _task.done()


async def stop() -> None:
    global _task
    if _task is None:
        return
    _task.cancel()
    try:
        await _task
    except asyncio.CancelledError:
        pass
    _task = None
