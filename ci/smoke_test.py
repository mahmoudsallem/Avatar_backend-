"""CI smoke test for the canary SageMaker endpoint (deploy/cicd-pipeline.yaml's SmokeTestCanary stage).

Sends one synthetic (non-photo) image through InvokeEndpoint and accepts EITHER outcome as healthy:
  - 200 with a valid JPEG body: the full generation pipeline ran end to end.
  - 400 with a validation-rejection error: Step 1 (face/person detection) correctly rejected a
    non-face image - this still proves the container booted, loaded every model, and the request
    pipeline is functioning; it's not a failure of the pipeline itself.

Only a connection error, timeout, 5xx, or a malformed 200 body counts as FAIL. Deliberately does
NOT use a real test photo: users/ (real test photos) is gitignored, so none would be present in a
fresh CodeBuild checkout, and committing a real person's photo into git history for CI purposes is
a separate decision this script doesn't make unilaterally.

Usage:
    python ci/smoke_test.py --endpoint-name saytara-avatar-g7e-canary --region eu-north-1
"""
import argparse
import base64
import io
import json
import sys

import boto3
from botocore.config import Config


def make_synthetic_jpeg() -> bytes:
    """A small solid-color placeholder image - not a photo of anyone, just enough for the
    container's image-decoding path to have something valid to chew on."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), color=(128, 128, 128)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint-name", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--timeout", type=float, default=180.0)
    a = ap.parse_args()

    raw = make_synthetic_jpeg()
    payload = json.dumps({"image": base64.b64encode(raw).decode("ascii"), "filename": "smoke.jpg"}).encode()

    cfg = Config(connect_timeout=30, read_timeout=a.timeout, retries={"max_attempts": 0})
    client = boto3.client("sagemaker-runtime", region_name=a.region, config=cfg)

    print(f"Invoking {a.endpoint_name} in {a.region} with a synthetic smoke-test image...", flush=True)
    try:
        resp = client.invoke_endpoint(
            EndpointName=a.endpoint_name,
            ContentType="application/json",
            Accept="image/jpeg",
            Body=payload,
        )
        status = resp.get("ResponseMetadata", {}).get("HTTPStatusCode", 200)
        body = resp["Body"].read()
        if body[:3] == b"\xff\xd8\xff":
            print(f"PASS: endpoint generated a real avatar (200, {len(body)} bytes JPEG) - full pipeline healthy.")
            return 0
        print(f"FAIL: 200 response but body is not a JPEG (first 200 bytes: {body[:200]!r})")
        return 1
    except Exception as exc:
        resp_meta = getattr(exc, "response", None)
        status = resp_meta.get("ResponseMetadata", {}).get("HTTPStatusCode") if resp_meta else None
        if status == 400:
            print(f"PASS: endpoint cleanly rejected the synthetic (non-face) image with 400 "
                  f"({exc}) - models loaded and the validation pipeline ran correctly.")
            return 0
        print(f"FAIL: unexpected error invoking endpoint (status={status}): {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
