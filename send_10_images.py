"""Fire N photos (default 10) at the avatar backend AT THE SAME TIME and report whether it works.

    python send_10_images.py --url http://<ec2-ip>:8000                # sends EVERY photo in users/ at once
    python send_10_images.py --url http://<ec2-ip>:8000 --count 20     # or a fixed number (photos are cycled)

For every request it logs: input photo (name, size), output avatar (path, size), start / end clock
time, duration, server generation time, tries and identity score. Everything is written to
./test_results/<run-time>/ :
    inputs/   copy of each input photo        outputs/  each returned avatar
    run.log   full log                        results.csv   one row per request
    report.html   input and output side by side (open it in a browser)
If GPU_SLOTS > 1 works, wall-clock should be clearly LESS than the sum of the per-image times.
Only needs the Python standard library.
"""
import argparse
import base64
import csv
import html
import json
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LOG = None
LOG_LOCK = threading.Lock()
RUNNING = {}          # idx -> (input name, start time) for requests still waiting for the server
DONE = []             # finished request numbers


def log(msg=""):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}" if msg else ""
    with LOG_LOCK:
        print(line, flush=True)
        if LOG:
            LOG.write(line + "\n")
            LOG.flush()


def kb(n):
    return f"{n / 1024:.0f} KB"


def get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def send(endpoint, img, idx, out_dir, timeout, t_run):
    in_copy = out_dir / "inputs" / f"{idx:02d}_{img.name}"
    shutil.copyfile(img, in_copy)
    raw = img.read_bytes()
    payload = json.dumps({"image": base64.b64encode(raw).decode("ascii"), "filename": img.name}).encode()
    req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    row = {"n": idx, "input": img.name, "input_kb": round(len(raw) / 1024), "input_copy": in_copy.relative_to(out_dir).as_posix(),
           "output": "", "output_kb": 0, "status": None, "ok": False, "rejected": False,
           "start": "", "end": "", "start_s": 0.0, "end_s": 0.0, "sec": 0.0,
           "gen": "", "tries": "", "best_try": "", "id_sim": "", "visor": "", "note": ""}
    t0 = time.time()
    row["start"] = datetime.now().strftime("%H:%M:%S")
    row["start_s"] = round(t0 - t_run, 1)
    RUNNING[idx] = (img.name, t0)
    log(f"#{idx:02d} SENT      input={img.name} ({kb(len(raw))})")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            row["status"] = r.status
            h = r.headers
            row["gen"] = h.get("X-Generation-Seconds", "")
            row["tries"] = h.get("X-Tries-Used", "")
            row["best_try"] = h.get("X-Best-Try", "")
            row["id_sim"] = h.get("X-Identity-Similarity", "")
            row["visor"] = h.get("X-Visor-Status", "")
            if r.status == 200 and body[:3] == b"\xff\xd8\xff":
                op = out_dir / "outputs" / f"{idx:02d}_{Path(img.name).stem}_avatar.jpg"
                op.write_bytes(body)
                row.update(ok=True, output=op.relative_to(out_dir).as_posix(), output_kb=round(len(body) / 1024))
            else:
                row["note"] = "200 but not a JPEG"
    except urllib.error.HTTPError as e:
        row["status"] = e.code
        row["rejected"] = e.code == 400      # validation rejected the photo: server worked, photo unusable
        row["note"] = e.read().decode("utf-8", "ignore")[:200]
    except Exception as e:                    # timeout, connection error, ...
        row["note"] = str(e)[:200]
    t1 = time.time()
    row["end"] = datetime.now().strftime("%H:%M:%S")
    row["end_s"] = round(t1 - t_run, 1)
    row["sec"] = round(t1 - t0, 1)
    RUNNING.pop(idx, None)
    DONE.append(idx)
    res = "OK" if row["ok"] else ("REJECTED" if row["rejected"] else "FAILED")
    log(f"#{idx:02d} {res:<9} input={img.name} -> output={row['output'] or '-'}"
        f"{' (' + str(row['output_kb']) + ' KB)' if row['ok'] else ''}  "
        f"took {row['sec']}s (server gen {row['gen'] or '-'}s, tries={row['tries'] or '-'}, id_sim={row['id_sim'] or '-'}, "
        f"visor={row['visor'] or '-'}){'  ' + row['note'] if row['note'] else ''}")
    return row


def heartbeat(base, total, t_run, stop, interval):
    """Every few seconds: how many are done / still running, how long each has waited, and the GPU memory."""
    while not stop.wait(interval):
        try:
            h = get_json(base + "/health", timeout=3)
            gpu = f"VRAM free {h.get('gpu_free_vram_gb')} GB"
        except Exception:
            gpu = "VRAM n/a"
        now = time.time()
        waiting = ", ".join(f"#{i:02d} {int(now - s)}s" for i, (_, s) in sorted(RUNNING.items()))
        log(f"... {int(now - t_run)}s elapsed | done {len(DONE)}/{total} | running {len(RUNNING)} | {gpu} | waiting: {waiting or '-'}")


def write_report(out_dir, rows, wall, meta):
    cards = []
    for r in rows:
        out = (f'<img src="{html.escape(r["output"])}">' if r["ok"] else
               f'<div class="bad">{"REJECTED" if r["rejected"] else "FAILED"}<br>{html.escape(r["note"])}</div>')
        cards.append(
            f'<div class="card"><div class="t">#{r["n"]:02d} &middot; {html.escape(r["input"])}</div>'
            f'<div class="pair"><figure><img src="{html.escape(r["input_copy"])}"><figcaption>input ({r["input_kb"]} KB)</figcaption></figure>'
            f'<figure>{out}<figcaption>output ({r["output_kb"]} KB)</figcaption></figure></div>'
            f'<div class="m">start {r["start"]} (+{r["start_s"]}s) &rarr; end {r["end"]} (+{r["end_s"]}s) &middot; '
            f'<b>{r["sec"]}s</b> &middot; server gen {r["gen"] or "-"}s &middot; tries {r["tries"] or "-"} &middot; id_sim {r["id_sim"] or "-"} &middot; visor {r["visor"] or "-"}</div></div>')
    page = (f'<!doctype html><meta charset="utf-8"><title>Avatar test report</title>'
            '<style>body{font:14px system-ui;background:#111;color:#eee;margin:20px}.card{background:#1c1c1c;border-radius:10px;padding:12px;margin:12px 0}'
            '.t{font-weight:600;margin-bottom:8px}.pair{display:flex;gap:12px}figure{margin:0}img{max-height:320px;max-width:45vw;border-radius:6px}'
            'figcaption,.m{color:#aaa;font-size:12px;margin-top:4px}.bad{padding:30px;background:#3a1a1a;border-radius:6px}</style>'
            f'<h2>Avatar concurrency test</h2><p>{html.escape(meta)}<br>wall-clock <b>{wall}s</b></p>' + "".join(cards))
    (out_dir / "report.html").write_text(page, encoding="utf-8")


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--count", type=int, default=0, help="how many requests to fire at once; 0 (default) = send EVERY photo in the folder, no limit")
    ap.add_argument("--users-dir", default="users", help="folder with test photos")
    ap.add_argument("--out", default="test_results")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between live progress lines")
    a = ap.parse_args()

    out_dir = Path(a.out) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "outputs").mkdir(exist_ok=True)
    LOG = open(out_dir / "run.log", "w", encoding="utf-8")

    base = a.url.rstrip("/")
    try:
        h = get_json(base + "/health")
    except Exception as e:
        log(f"Cannot reach {base}/health: {e}")
        sys.exit(2)
    meta = (f"server {base} | status={h.get('status')} gpu={h.get('device_name')} gpu_slots={h.get('gpu_slots', '?')} "
            f"free_vram={h.get('gpu_free_vram_gb')}GB")
    log(meta)
    if h.get("status") != "ok":
        log("Server is not healthy - fix that first.")
        sys.exit(2)

    d = Path(a.users_dir)
    imgs = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTS) if d.is_dir() else []
    if not imgs:
        log(f"No photos in {d.resolve()} - copy some test photos there first.")
        sys.exit(2)
    batch = list(imgs) if a.count <= 0 else [imgs[i % len(imgs)] for i in range(a.count)]   # default: all photos
    meta += f" | {len(batch)} requests at once"
    log(f"Sending {len(batch)} photos at the same time to {base}/v1/avatar/base64 ...")
    log()

    t_run = time.time()
    rows = []
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(base, len(batch), t_run, stop, a.interval), daemon=True).start()
    with ThreadPoolExecutor(max_workers=len(batch)) as pool:
        futs = [pool.submit(send, base + "/v1/avatar/base64", img, i + 1, out_dir, a.timeout, t_run) for i, img in enumerate(batch)]
        for f in as_completed(futs):
            rows.append(f.result())
    stop.set()
    rows.sort(key=lambda r: r["n"])
    wall = round(time.time() - t_run, 1)

    log()
    log(f"{'#':>2}  {'input':<30} {'output':<34} {'start':>8} {'end':>8} {'total s':>8} {'gen s':>6} {'tries':>5} {'id_sim':>7}  result")
    for r in rows:
        res = "OK" if r["ok"] else ("REJECTED" if r["rejected"] else "FAILED")
        log(f"{r['n']:>2}  {r['input'][:30]:<30} {(Path(r['output']).name if r['output'] else '-')[:34]:<34} "
            f"{r['start']:>8} {r['end']:>8} {r['sec']:>8} {r['gen']:>6} {r['tries']:>5} {r['id_sim']:>7}  {res}")

    ok = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"] and not r["rejected"]]
    gens = [float(r["gen"]) for r in ok if r["gen"]]
    log()
    log(f"Generated {len(ok)}/{len(rows)} | failed: {len(bad)} | rejected photos: {sum(r['rejected'] for r in rows)} | "
        f"wall-clock: {wall}s | throughput: {len(ok) / wall * 60:.1f} avatars/min")
    if ok:
        secs = [r["sec"] for r in ok]
        log(f"Per-avatar time: min {min(secs)}s | avg {sum(secs) / len(secs):.1f}s | max {max(secs)}s")
    if ok:
        log(f"Effective time per avatar: {wall / len(ok):.1f}s (wall-clock / avatars) -> about {len(ok) / wall * 3600:.0f} avatars per hour")
        log("Note: per-avatar 'total s' is long because all requests run at the same time and share the GPU; compare the wall-clock between runs.")
    tries1 = [r for r in ok if str(r["tries"]) == "1"]
    if ok and len(tries1) == len(ok):
        log("All avatars used only 1 try - if BEST_OF_N > 1, TIME_BUDGET is probably shorter than one try under load (set TIME_BUDGET=0 to test).")

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_report(out_dir, rows, wall, meta)
    log(f"Saved to {out_dir.resolve()}  (open report.html to see input and output side by side)")
    LOG.close()
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
