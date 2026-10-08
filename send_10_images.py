"""Fire N photos (default 10) at the avatar backend AT THE SAME TIME and report whether it works.

    python send_10_images.py --url http://localhost:8000
    python send_10_images.py --url http://<ec2-ip>:8000 --count 10 --users-dir users

Saves every returned avatar to ./test_results/ and prints a per-image table plus the total
wall-clock time. If GPU_SLOTS > 1 works, wall-clock should be clearly LESS than the sum of the
per-image times (that sum is what a one-at-a-time server would need).
Only needs the Python standard library.
"""
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def send(endpoint, img, idx, out_dir, timeout):
    payload = json.dumps({
        "image": base64.b64encode(img.read_bytes()).decode("ascii"),
        "filename": img.name,
    }).encode()
    req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    row = {"n": idx, "file": img.name, "status": None, "ok": False, "rejected": False,
           "sec": 0.0, "gen": "", "tries": "", "id_sim": "", "note": ""}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            row["status"] = r.status
            row["gen"] = r.headers.get("X-Generation-Seconds", "")
            row["tries"] = r.headers.get("X-Tries-Used", "")
            row["id_sim"] = r.headers.get("X-Identity-Similarity", "")
            if r.status == 200 and body[:3] == b"\xff\xd8\xff":
                row["ok"] = True
                (out_dir / f"{idx:02d}_{Path(img.name).stem}_avatar.jpg").write_bytes(body)
            else:
                row["note"] = "200 but not a JPEG"
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")[:120]
        row["status"] = e.code
        row["rejected"] = e.code == 400          # photo rejected by validation = server worked, photo not usable
        row["note"] = msg
    except Exception as e:                        # timeouts, connection errors, ...
        row["note"] = str(e)[:120]
    row["sec"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--count", type=int, default=10, help="how many requests to fire at once")
    ap.add_argument("--users-dir", default="users", help="folder with test photos (cycled if fewer than --count)")
    ap.add_argument("--out", default="test_results")
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args()

    base = a.url.rstrip("/")
    try:
        h = get_json(base + "/health")
    except Exception as e:
        print(f"Cannot reach {base}/health: {e}")
        sys.exit(2)
    print(f"Server: status={h.get('status')} gpu={h.get('device_name')} gpu_slots={h.get('gpu_slots', '?')} "
          f"free_vram={h.get('gpu_free_vram_gb')}GB")
    if h.get("status") != "ok":
        print("Server is not healthy - fix that first.")
        sys.exit(2)

    d = Path(a.users_dir)
    imgs = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTS) if d.is_dir() else []
    if not imgs:
        print(f"No photos in {d.resolve()} - copy some test photos there first.")
        sys.exit(2)
    batch = [imgs[i % len(imgs)] for i in range(a.count)]
    out_dir = Path(a.out)
    out_dir.mkdir(exist_ok=True)

    print(f"Sending {len(batch)} photos at the same time to {base}/v1/avatar/base64 ...\n")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(batch)) as pool:
        rows = list(pool.map(lambda x: send(base + "/v1/avatar/base64", x[1], x[0] + 1, out_dir, a.timeout),
                             enumerate(batch)))
    wall = round(time.time() - t0, 1)

    print(f"{'#':>2}  {'file':<34} {'HTTP':>4} {'total s':>8} {'gen s':>6} {'tries':>5} {'id_sim':>7}  result")
    for r in rows:
        res = "OK" if r["ok"] else ("REJECTED (bad photo)" if r["rejected"] else "FAILED")
        print(f"{r['n']:>2}  {r['file'][:34]:<34} {str(r['status']):>4} {r['sec']:>8} {r['gen']:>6} "
              f"{r['tries']:>5} {r['id_sim']:>7}  {res} {r['note'] if not r['ok'] else ''}")

    ok = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"] and not r["rejected"]]
    gens = [float(r["gen"]) for r in ok if r["gen"]]
    print(f"\nGenerated {len(ok)}/{len(rows)}  |  failed: {len(bad)}  |  rejected photos: "
          f"{sum(r['rejected'] for r in rows)}  |  wall-clock: {wall}s  |  throughput: {len(ok) / wall * 60:.1f} avatars/min")
    if gens:
        seq = round(sum(gens), 1)
        print(f"One-at-a-time would need about {seq}s (sum of per-image times) -> speed-up x{seq / wall:.2f}")
        if seq / wall < 1.15:
            print("   -> little/no overlap: check the log for 'GPU slots ready: N' (N should be > 1).")
    print(f"Avatars saved in {out_dir.resolve()}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
