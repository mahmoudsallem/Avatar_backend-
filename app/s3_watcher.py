"""
Picks up new user photos from s3://<S3_BUCKET>/<S3_USERS_PREFIX>, runs each through the same
analysis + generation pipeline as the HTTP endpoints, and uploads the resulting avatar to
s3://<S3_BUCKET>/<S3_GENERATED_PREFIX> under the same base filename (as a .jpg - the pipeline's
output is always JPEG). The source photo is then deleted from S3_USERS_PREFIX, so it behaves
like a move rather than a copy.

Two modes, picked automatically by whether S3_QUEUE_URL is set:

- Real-time (S3_QUEUE_URL set): an S3 "ObjectCreated" event notification on Users/* feeds an SQS
  queue, and this process long-polls that queue - a new photo is picked up within ~1-2s instead
  of waiting on a timer, and there's no need to repeatedly list the whole prefix. A message is
  only deleted once its photo is fully handled (generated, rejected, or already done); a transient
  failure is left alone so SQS redelivers it, bounded by the queue's maxReceiveCount before it
  lands in the dead-letter queue.
- Polling (S3_QUEUE_URL empty): lists Users/ every S3_POLL_INTERVAL seconds. Used when no SQS
  queue is set up (e.g. local/dev runs).

Either way, a photo counts as "already done" when a same-named file already exists in
S3_GENERATED_PREFIX, so the watcher is safe to restart (or redeliver a message) without redoing
work or needing a separate database of what's processed - this also covers the rare case where
the avatar was generated and uploaded but the source delete failed, leaving the photo behind in
Users/.

Enable with S3_ENABLED=true in .env. AWS credentials come from the normal boto3 chain (env vars,
~/.aws/credentials, or an EC2 instance role).
"""
import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings
from app import pipeline

logger = logging.getLogger("saytara.s3_watcher")

_s3 = None
_sqs = None
_task: Optional[asyncio.Task] = None
# Polling mode only: stems rejected this run - skipped until the process restarts, so a bad photo
# doesn't get retried (and burn a GPU slot) on every poll. The SQS path needs no equivalent - a
# rejected photo's message is deleted immediately, so it simply won't come around again.
_failed_stems: Set[str] = set()


def _client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=settings.S3_REGION)
    return _s3


def _sqs_client():
    global _sqs
    if _sqs is None:
        _sqs = boto3.client("sqs", region_name=settings.S3_REGION)
    return _sqs


async def _process_one(s3, key: str) -> str:
    """Process one Users/<key> photo.

    Returns one of:
      "done"         - generated and uploaded successfully
      "already_done" - a Generated_Avatar/ output already existed (e.g. duplicate delivery) - skipped
      "rejected"     - Step 1 validation permanently rejected the photo - don't retry
      "retry"        - a transient failure - should be retried later
    """
    stem = Path(key).stem
    out_key = f"{settings.S3_GENERATED_PREFIX}{stem}.jpg"

    try:
        await asyncio.to_thread(s3.head_object, Bucket=settings.S3_BUCKET, Key=out_key)
        return "already_done"
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
            logger.exception("S3 watcher: could not check s3://%s/%s", settings.S3_BUCKET, out_key)
            return "retry"
    except BotoCoreError:
        logger.exception("S3 watcher: could not check s3://%s/%s", settings.S3_BUCKET, out_key)
        return "retry"

    settings.VAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = settings.VAL_TMP_DIR / f"s3_{uuid.uuid4().hex}{Path(key).suffix.lower()}"
    try:
        await asyncio.to_thread(s3.download_file, settings.S3_BUCKET, key, str(tmp_path))
        jpeg_bytes, metadata = await pipeline.process_image_file(tmp_path)

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

        try:
            await asyncio.to_thread(s3.delete_object, Bucket=settings.S3_BUCKET, Key=key)
            logger.info("S3 watcher: removed processed photo s3://%s/%s", settings.S3_BUCKET, key)
        except (BotoCoreError, ClientError):
            # The avatar is already generated and uploaded - the head_object check above keeps
            # this from being reprocessed even if the source photo lingers in Users/.
            logger.exception("S3 watcher: generated OK but could not delete source s3://%s/%s", settings.S3_BUCKET, key)
        return "done"

    except pipeline.Rejected as r:
        logger.warning("S3 watcher: rejected s3://%s/%s: %s", settings.S3_BUCKET, key, r)
        return "rejected"
    except (BotoCoreError, ClientError):
        logger.exception("S3 watcher: S3 error handling s3://%s/%s", settings.S3_BUCKET, key)
        return "retry"
    except Exception:
        logger.exception("S3 watcher: failed to process s3://%s/%s", settings.S3_BUCKET, key)
        return "retry"
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


# ============================================================================
# Real-time mode: SQS long-polling (S3_QUEUE_URL set)
# ============================================================================

async def _process_batch(sqs, sem: asyncio.Semaphore, messages: List[dict]) -> None:
    """Handle one received batch of SQS messages as a unit: flatten every photo across the whole
    batch, sort by the photo's actual S3 upload time (not SQS delivery order, which isn't
    guaranteed to match), then run them through the shared concurrency limit in that order - so
    a burst of uploads is processed oldest-first while still running S3_MAX_CONCURRENT at once."""
    # (event_time, key, message_index) for every real photo across the batch
    items: List[tuple] = []
    # A message is deleted only once every photo it carried is fully handled (done/already_done/
    # rejected); a "retry" anywhere in it leaves the whole message for SQS to redeliver.
    message_ok = [True] * len(messages)

    for idx, message in enumerate(messages):
        try:
            body = json.loads(message["Body"])
        except Exception:
            logger.warning("S3 watcher: unparseable SQS message, discarding: %r", message.get("Body"))
            continue  # message_ok stays True - it gets deleted below, nothing can be done with it

        records = body.get("Records") or []
        if not records:
            # S3's bucket-notification test event (sent once when the notification is configured),
            # or an unrelated message - nothing to process, but still fine to delete.
            continue

        for rec in records:
            key = unquote_plus(rec["s3"]["object"]["key"])
            if key.endswith("/") or Path(key).suffix.lower() not in pipeline.IMAGE_EXTENSIONS:
                continue
            items.append((rec.get("eventTime", ""), key, idx))

    items.sort(key=lambda item: item[0])  # S3 eventTime is ISO 8601 - sorts chronologically as text
    if items:
        logger.info(
            "S3 watcher: processing %d photo(s) from this batch in upload order (oldest first)",
            len(items),
        )

    s3 = _client()

    async def _run(event_time: str, key: str, idx: int) -> None:
        async with sem:
            result = await _process_one(s3, key)
        if result == "retry":
            message_ok[idx] = False

    await asyncio.gather(*(_run(*item) for item in items))

    for idx, message in enumerate(messages):
        if message_ok[idx]:
            await asyncio.to_thread(sqs.delete_message, QueueUrl=settings.S3_QUEUE_URL, ReceiptHandle=message["ReceiptHandle"])
        # else: leave it - its visibility timeout expires and SQS redelivers it, bounded by the
        # queue's maxReceiveCount before it lands in the dead-letter queue.


async def _sqs_watch_loop() -> None:
    sqs = _sqs_client()
    logger.info(
        "S3 watcher started (real-time via SQS, oldest-upload-first within each batch): %s -> s3://%s/%s",
        settings.S3_QUEUE_URL, settings.S3_BUCKET, settings.S3_GENERATED_PREFIX,
    )
    sem = asyncio.Semaphore(settings.S3_MAX_CONCURRENT)
    while True:
        try:
            resp = await asyncio.to_thread(
                sqs.receive_message,
                QueueUrl=settings.S3_QUEUE_URL,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=settings.S3_QUEUE_WAIT_SECONDS,
            )
            messages = resp.get("Messages", [])
            if not messages:
                continue
            logger.info("S3 watcher: %d SQS message(s) received", len(messages))
            await _process_batch(sqs, sem, messages)
        except (BotoCoreError, ClientError):
            logger.exception("S3 watcher: SQS error during receive")
            await asyncio.sleep(5)
        except Exception:
            logger.exception("S3 watcher: unexpected error in SQS loop")
            await asyncio.sleep(5)


# ============================================================================
# Fallback mode: periodic listing (S3_QUEUE_URL empty)
# ============================================================================

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


async def _poll_once() -> None:
    s3 = _client()
    keys = await asyncio.to_thread(_pending_keys, s3)
    if not keys:
        return

    logger.info("S3 watcher: %d new user photo(s) to process", len(keys))
    sem = asyncio.Semaphore(settings.S3_MAX_CONCURRENT)

    async def _guarded(key: str) -> None:
        async with sem:
            result = await _process_one(s3, key)
            if result == "rejected":
                _failed_stems.add(Path(key).stem)

    await asyncio.gather(*(_guarded(key) for key in keys))


async def _poll_watch_loop() -> None:
    logger.info(
        "S3 watcher started (polling every %.0fs): s3://%s/%s -> s3://%s/%s",
        settings.S3_POLL_INTERVAL,
        settings.S3_BUCKET, settings.S3_USERS_PREFIX,
        settings.S3_BUCKET, settings.S3_GENERATED_PREFIX,
    )
    while True:
        try:
            await _poll_once()
        except (BotoCoreError, ClientError):
            logger.exception("S3 watcher: S3 error during poll")
        except Exception:
            logger.exception("S3 watcher: unexpected error during poll")
        await asyncio.sleep(settings.S3_POLL_INTERVAL)


# ============================================================================
# Lifecycle
# ============================================================================

def start() -> Optional[asyncio.Task]:
    """Start the background watch loop if S3_ENABLED; no-op otherwise. Uses SQS (real-time) when
    S3_QUEUE_URL is set, otherwise falls back to periodic polling."""
    global _task
    if not settings.S3_ENABLED:
        logger.info("S3 watcher disabled (S3_ENABLED=false)")
        return None
    loop = _sqs_watch_loop() if settings.S3_QUEUE_URL else _poll_watch_loop()
    _task = asyncio.create_task(loop)
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
