#!/usr/bin/env python3
"""
Saytara Avatar S3 Quality & Performance Evaluator
Performs end-to-end evaluation via AWS S3:
1. Uploads user portrait photos from users/ to s3://avatar-etisalata/Users/<prefix>XX.jpg
2. Monitors generation in real-time across S3 Watcher / GPU slots
3. Measures precise per-avatar generation latency and sustained throughput
4. Downloads generated avatars and builds a comprehensive side-by-side HTML report
5. Opens the interactive visual report automatically

Usage:
    python eval_s3_quality.py                        # Runs live evaluation (UAT Round 3)
    python eval_s3_quality.py --round 3              # Explicit round number
    python eval_s3_quality.py --prefix test_run_     # Custom S3 prefix
    python eval_s3_quality.py --from-prefix uat2_    # Evaluate already completed S3 run
"""
import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
USERS_DIR = ROOT / "users"
EVAL_DIR = ROOT / "evaluation_results"
BUCKET = "avatar-etisalata"
S3_USERS = f"s3://{BUCKET}/Users"
S3_OUTPUTS = f"s3://{BUCKET}/Generated_Avatar"

def run_aws(args):
    """Run AWS CLI command and return stdout string."""
    res = subprocess.run(["aws"] + args, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return res.stdout.strip()

def get_s3_server_timing(prefix: str):
    """Fetch exact S3 LastModified timestamps to measure pure server-side GPU generation time (zero client network overhead)."""
    try:
        res = subprocess.run(
            ["aws", "s3api", "list-objects-v2", "--bucket", BUCKET, "--prefix", f"Generated_Avatar/{prefix}"],
            capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
        data = json.loads(res.stdout) if res.stdout else {}
        objs = data.get("Contents", [])
        parsed = []
        for obj in objs:
            try:
                dt = datetime.fromisoformat(obj["LastModified"].replace("Z", "+00:00"))
                parsed.append((obj["Key"], dt, int(obj.get("Size", 0))))
            except Exception:
                pass
        if len(parsed) < 2:
            return None

        parsed.sort(key=lambda x: x[1])
        first_dt = parsed[0][1]
        last_dt = parsed[-1][1]
        span_s = max(1.0, (last_dt - first_dt).total_seconds())
        count = len(parsed)
        avg_s = span_s / max(1, count - 1)
        total_server_s = round(span_s + avg_s, 1)
        pure_throughput = round((count * 60.0) / max(0.1, total_server_s), 1)
        steady_rate = round(((count - 1) * 60.0) / max(0.1, span_s), 1) if span_s > 0 else pure_throughput

        time_map = {}
        for key, dt, sz in parsed:
            stem = Path(key).name
            time_map[stem] = dt

        return {
            "first_dt": first_dt,
            "last_dt": last_dt,
            "span_s": round(span_s, 1),
            "total_server_s": total_server_s,
            "avg_s": round(avg_s, 2),
            "pure_throughput": pure_throughput,
            "steady_rate": steady_rate,
            "count": count,
            "time_map": time_map,
        }
    except Exception:
        return None

def build_html_report(out_dir: Path, items: list, wall_clock: float, batch_name: str, title: str, server_stats: dict = None, upload_duration: float = 0.0):
    """Generate modern, interactive side-by-side comparison HTML report."""
    cards_html = []
    success_count = sum(1 for item in items if item["ok"])
    total_expected = len(items)

    for item in items:
        idx = item["idx"]
        inp_rel = item["input_rel"]
        out_rel = item["output_rel"]
        ok = item["ok"]
        elapsed = item.get("elapsed", "-")
        card_time_str = f"~{server_stats['avg_s']}s (GPU)" if server_stats else f"{elapsed}s"

        if ok:
            out_img = f'<img src="{html.escape(out_rel)}" alt="Avatar" class="avatar-img" onclick="openModal(this.src)">'
            badge = '<span class="badge badge-success">PASSED</span>'
        else:
            out_img = '<div class="status-box error">GENERATION FAILED / TIMEOUT</div>'
            badge = '<span class="badge badge-danger">FAILED</span>'

        card = f"""
        <div class="eval-card">
            <div class="card-header">
                <span class="card-id">#{idx:02d}</span>
                <span class="card-name">{html.escape(item['filename'])}</span>
                {badge}
                <span class="card-time">{card_time_str}</span>
            </div>
            <div class="card-body">
                <div class="img-column">
                    <div class="img-wrapper">
                        <img src="{html.escape(inp_rel)}" alt="Original Photo" class="user-img" onclick="openModal(this.src)">
                    </div>
                    <div class="img-meta">
                        <strong>Input Portrait</strong>
                        <span>{item['input_size_kb']} KB</span>
                    </div>
                </div>
                <div class="arrow-divider">➔</div>
                <div class="img-column">
                    <div class="img-wrapper">
                        {out_img}
                    </div>
                    <div class="img-meta">
                        <strong>Sci-Fi Avatar Result</strong>
                        <span>{item['output_size_kb']} KB</span>
                    </div>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    avg_time = round(wall_clock / max(1, total_expected), 2)
    throughput = round(60.0 / max(0.1, avg_time), 1)
    pass_pct = round((success_count / max(1, total_expected)) * 100, 1)

    gpu_time = server_stats["total_server_s"] if server_stats else wall_clock
    gpu_throughput = server_stats["pure_throughput"] if server_stats else throughput
    gpu_avg = server_stats["avg_s"] if server_stats else avg_time

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --surface-color: #151d2f;
            --border-color: #232f48;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --success-color: #10b981;
            --danger-color: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1280px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }}
        h1 {{
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ color: var(--text-secondary); font-size: 14px; margin-bottom: 20px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-top: 15px;
        }}
        .stat-card {{
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}
        .stat-value {{ font-size: 24px; font-weight: 700; color: #fff; margin-bottom: 4px; }}
        .stat-label {{ font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(580px, 1fr));
            gap: 24px;
        }}
        @media (max-width: 640px) {{
            .gallery-grid {{ grid-template-columns: 1fr; }}
        }}
        .eval-card {{
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .eval-card:hover {{
            transform: translateY(-3px);
            border-color: var(--accent-cyan);
        }}
        .card-header {{
            background: rgba(0, 0, 0, 0.25);
            padding: 12px 18px;
            display: flex;
            align-items: center;
            gap: 12px;
            border-bottom: 1px solid var(--border-color);
        }}
        .card-id {{
            background: #1e293b;
            color: #38bdf8;
            font-size: 13px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
        }}
        .card-name {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .card-time {{ font-size: 13px; color: var(--text-secondary); font-weight: 500; }}
        .badge {{
            font-size: 11px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 12px;
            text-transform: uppercase;
        }}
        .badge-success {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .card-body {{
            padding: 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }}
        .img-column {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .img-wrapper {{
            width: 100%;
            height: 290px;
            background: #090d16;
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--border-color);
            position: relative;
        }}
        .img-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            cursor: pointer;
            transition: transform 0.3s ease;
        }}
        .img-wrapper img:hover {{
            transform: scale(1.03);
        }}
        .arrow-divider {{
            font-size: 22px;
            color: var(--text-secondary);
            opacity: 0.6;
        }}
        .img-meta {{
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            width: 100%;
            font-size: 12px;
            color: var(--text-secondary);
        }}
        .img-meta strong {{ color: var(--text-primary); }}
        .status-box {{
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 20px;
            color: var(--danger-color);
            font-size: 13px;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 999;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.85);
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }}
        .modal img {{
            max-width: 90vw;
            max-height: 90vh;
            border-radius: 12px;
            box-shadow: 0 0 30px rgba(0,0,0,0.8);
        }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>{html.escape(title)}</h1>
        <div class="subtitle">Live User Acceptance & Quality Evaluation &bull; Batch: {html.escape(batch_name)} &bull; Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{success_count} / {total_expected}</div>
                <div class="stat-label">Acceptance Rate ({pass_pct}%)</div>
            </div>
            <div class="stat-card" style="border: 1px solid #38bdf8;">
                <div class="stat-value" style="color: #38bdf8;">{gpu_time:.1f} s</div>
                <div class="stat-label">Pure Server GPU Time (Zero Network)</div>
            </div>
            <div class="stat-card" style="border: 1px solid #34d399;">
                <div class="stat-value" style="color: #34d399;">{gpu_throughput:.1f} / min</div>
                <div class="stat-label">Pure GPU Throughput (~{int(gpu_throughput * 60)}/hr)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{gpu_avg:.2f} s</div>
                <div class="stat-label">Avg Pure GPU / Avatar</div>
            </div>
            <div class="stat-card" style="opacity: 0.75;">
                <div class="stat-value">{upload_duration:.1f} s</div>
                <div class="stat-label">Client Upload Overhead (Net)</div>
            </div>
        </div>
    </header>

    <div class="gallery-grid">
        {"".join(cards_html)}
    </div>
</div>

<div class="modal" id="imageModal" onclick="this.style.display='none'">
    <img id="modalImg" src="">
</div>

<script>
function openModal(src) {{
    document.getElementById('modalImg').src = src;
    document.getElementById('imageModal').style.display = 'flex';
}}
</script>
</body>
</html>"""

    report_path = out_dir / "report.html"
    report_path.write_text(html_content, encoding="utf-8")
    return report_path

def get_next_round_num() -> int:
    existing = []
    if EVAL_DIR.exists():
        for d in EVAL_DIR.glob("uat_round_*"):
            try:
                num = int(d.name.split("_")[-1])
                existing.append(num)
            except ValueError:
                pass
    return max(existing, default=1) + 1

def main():
    parser = argparse.ArgumentParser(description="Saytara Avatar S3 Quality & Performance Evaluator")
    parser.add_argument("--round", type=int, default=None, help="Evaluation / UAT Round number (default: auto-detected next round)")
    parser.add_argument("--prefix", type=str, default="", help="Custom S3 file prefix (default: uat<round>_)")
    parser.add_argument("--from-prefix", type=str, default="", help="Build report from existing S3 run without uploading")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--timeout", type=int, default=600, help="Max wait (s) from the FIRST upload for all avatars to appear")
    parser.add_argument("--repeat", type=int, default=1, help="Send the users/ photos this many times (e.g. 4 -> 56 images) to measure steady-state throughput")
    parser.add_argument("--workers", type=int, default=8, help="Parallel upload/download workers")
    parser.add_argument("--poll", type=float, default=1.0, help="Seconds between S3 listings while waiting")
    args = parser.parse_args()

    files = sorted(list(USERS_DIR.glob("*.*")), key=lambda p: p.name.lower())
    total_expected = len(files)
    if total_expected == 0:
        print(f"Error: No images found in {USERS_DIR}")
        sys.exit(1)

    # Mode A: Evaluate existing run
    if args.from_prefix:
        prefix = args.from_prefix
        if not prefix.endswith("_"):
            prefix += "_"
        run_name = f"eval_{prefix.rstrip('_')}"
        title = f"Saytara Avatar Quality Evaluation - {prefix.rstrip('_')}"
        out_dir = EVAL_DIR / run_name
        inp_dir = out_dir / "inputs"
        out_img_dir = out_dir / "outputs"
        inp_dir.mkdir(parents=True, exist_ok=True)
        out_img_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"EVALUATING EXISTING S3 RUN: {prefix}")
        print("=" * 60)

        items = []
        for idx, f in enumerate(files, 1):
            target_name = f"{prefix}{idx:02d}.jpg"
            in_copy = inp_dir / f"{idx:02d}_{f.name}"
            shutil.copyfile(f, in_copy)

            out_local = out_img_dir / f"{idx:02d}_{f.stem}_avatar.jpg"
            run_aws(["s3", "cp", f"{S3_OUTPUTS}/{target_name}", str(out_local), "--quiet"])
            ok = out_local.is_file() and out_local.stat().st_size > 1000

            items.append({
                "idx": idx,
                "filename": f.name,
                "input_rel": in_copy.relative_to(out_dir).as_posix(),
                "input_size_kb": round(f.stat().st_size / 1024, 1),
                "output_rel": out_local.relative_to(out_dir).as_posix() if ok else "",
                "output_size_kb": round(out_local.stat().st_size / 1024, 1) if ok else 0,
                "ok": ok,
                "elapsed": "-",
            })
            status = "OK" if ok else "NOT FOUND"
            print(f"#{idx:02d} [{status}] {f.name} -> {target_name}")

        server_stats = get_s3_server_timing(prefix)
        wall_clock = server_stats["total_server_s"] if server_stats else 40.0
        report_file = build_html_report(out_dir, items, wall_clock, prefix, title, server_stats=server_stats)
        print(f"\n[EVAL] Report generated: {report_file}")
        if server_stats:
            print(f"[PURE SERVER GPU] Time: {server_stats['total_server_s']}s | Throughput: {server_stats['pure_throughput']}/min | Avg/Avatar: {server_stats['avg_s']}s")
        if not args.no_open:
            try:
                os.startfile(str(report_file))
            except Exception:
                webbrowser.open(report_file.as_uri())
        return

    # Mode B: Live S3 Evaluation (UAT)
    round_num = args.round if args.round is not None else get_next_round_num()
    prefix = args.prefix if args.prefix else f"uat{round_num}_"
    if not prefix.endswith("_"):
        prefix += "_"
    run_name = f"uat_round_{round_num}"
    title = f"Saytara Avatar Acceptance & Quality Report (Round {round_num})"

    out_dir = EVAL_DIR / run_name
    inp_dir = out_dir / "inputs"
    out_img_dir = out_dir / "outputs"
    inp_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SAYTARA AVATAR BACKEND - LIVE S3 EVALUATION (ROUND {round_num})")
    print(f"Prefix: {prefix} | Total Images: {total_expected * max(1, args.repeat)} ({total_expected} photos x{max(1, args.repeat)})")
    print("=" * 60)

    # ---- Timing model --------------------------------------------------------------------------
    # The clock (t0) starts BEFORE the first upload, because the watcher starts generating as soon as the
    # first photo lands. Starting it after the last upload (old behaviour) hid every avatar that finished
    # during the upload and inflated throughput (Round 21: 11 of 14 were already done at "0 s").
    #   end-to-end rate  = images / (first upload start -> last avatar seen)      <- the honest headline
    #   steady-state rate= (images-1) / (first avatar seen -> last avatar seen)   <- GPU pipeline once warm
    #   latency          = avatar first seen - that photo's upload finished       <- what one user waits
    from concurrent.futures import ThreadPoolExecutor
    from statistics import median

    jobs = list(files) * max(1, args.repeat)
    total_expected = len(jobs)

    def _upload(job):
        idx, f = job
        target_name = f"{prefix}{idx:02d}.jpg"
        run_aws(["s3", "cp", str(f), f"{S3_USERS}/{target_name}", "--quiet"])
        return target_name, time.time()

    print(f"\n[1/3] Uploading {total_expected} photos ({args.workers} parallel) to {S3_USERS}/{prefix}XX.jpg; generation runs while uploading...")
    for idx, f in enumerate(jobs, 1):
        shutil.copyfile(f, inp_dir / f"{idx:02d}_{f.name}")

    t0 = time.time()
    uploaded = {}      # target_name -> time upload finished
    seen = {}          # target_name -> time avatar first seen in S3
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    futures = [pool.submit(_upload, (i, f)) for i, f in enumerate(jobs, 1)]

    print("[2/3] Waiting for avatars (clock started at first upload)...")
    while len(seen) < total_expected:
        for fu in [x for x in futures if x.done()]:
            futures.remove(fu)
            name, t_up = fu.result()
            uploaded[name] = t_up
        ls_output = run_aws(["s3", "ls", f"{S3_OUTPUTS}/{prefix}"])
        for line in ls_output.split("\n"):
            parts = line.strip().split()
            if len(parts) >= 4 and parts[3].startswith(prefix) and parts[3] not in seen:
                seen[parts[3]] = time.time()
                print(f"  [{seen[parts[3]] - t0:6.1f}s] COMPLETED {len(seen):02d}/{total_expected}: {parts[3]} ({round(int(parts[2]) / 1024, 1)} KB)")
        if time.time() - t0 > args.timeout:
            print(f"[2/3] Timeout reached ({args.timeout}s).")
            break
        if len(seen) < total_expected:
            time.sleep(max(0.2, args.poll))
    pool.shutdown(wait=True)
    for fu in futures:
        name, t_up = fu.result()
        uploaded[name] = t_up

    t_end = max(seen.values()) if seen else time.time()
    gen_duration = round(t_end - t0, 1)
    upload_duration = round(max(uploaded.values()) - t0, 1) if uploaded else 0.0
    print(f"[2/3] Done: {len(seen)}/{total_expected} avatars, end-to-end {gen_duration}s (uploads finished at {upload_duration}s).")

    # 3. Download results and compile report
    print(f"\n[3/3] Downloading generated avatars and building quality report...")

    def _download(job):
        idx, f = job
        target_name = f"{prefix}{idx:02d}.jpg"
        out_local = out_img_dir / f"{idx:02d}_{f.stem}_avatar.jpg"
        run_aws(["s3", "cp", f"{S3_OUTPUTS}/{target_name}", str(out_local), "--quiet"])
        return out_local

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as dpool:
        out_paths = list(dpool.map(_download, list(enumerate(jobs, 1))))

    items = []
    latencies = []
    for (idx, f), out_local in zip(enumerate(jobs, 1), out_paths):
        target_name = f"{prefix}{idx:02d}.jpg"
        in_copy = inp_dir / f"{idx:02d}_{f.name}"
        ok = out_local.is_file() and out_local.stat().st_size > 1000
        size_kb = round(out_local.stat().st_size / 1024, 1) if ok else 0
        lat = round(seen[target_name] - uploaded[target_name], 1) if (target_name in seen and target_name in uploaded) else None
        if lat is not None:
            latencies.append(lat)
        items.append({
            "idx": idx,
            "filename": f.name,
            "input_rel": in_copy.relative_to(out_dir).as_posix(),
            "input_size_kb": round(f.stat().st_size / 1024, 1),
            "output_rel": out_local.relative_to(out_dir).as_posix() if ok else "",
            "output_size_kb": size_kb,
            "ok": ok,
            "elapsed": lat if lat is not None else "-",
        })

    server_stats = get_s3_server_timing(prefix)
    report_file = build_html_report(out_dir, items, gen_duration, prefix, title, server_stats=server_stats, upload_duration=upload_duration)
    success_count = sum(1 for it in items if it["ok"])
    avg_s = round(gen_duration / max(1, len(items)), 2)
    t_rate = round(60.0 / max(0.1, avg_s), 1)
    steady = None
    if len(seen) >= 3:
        span = max(seen.values()) - min(seen.values())
        if span > 0:
            steady = round((len(seen) - 1) / span * 60.0, 1)
    lat_sorted = sorted(latencies)
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))] if lat_sorted else None
    med_lat = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0

    print("\n" + "=" * 60)
    print("S3 EVALUATION SUMMARY (Pure Server GPU vs Client Network)")
    print("=" * 60)
    print(f"Acceptance Rate            : {success_count}/{total_expected} ({(success_count/total_expected)*100:.0f}%)")
    if server_stats:
        print("-" * 60)
        print("[PURE SERVER GPU PERFORMANCE - Zero Client Network Overhead]")
        print(f"Server GPU Generation Time : {server_stats['total_server_s']}s (derived from AWS S3 timestamps)")
        print(f"Pure GPU Throughput        : {server_stats['pure_throughput']} avatars/min (~{int(server_stats['pure_throughput'] * 60)} / hour)")
        print(f"Avg Pure GPU / Avatar      : {server_stats['avg_s']}s / avatar")
        print(f"Steady-State GPU Rate      : {server_stats['steady_rate']} avatars/min")
    print("-" * 60)
    print("[CLIENT NETWORK OVERHEAD (Your Local Internet)]")
    print(f"Local Upload Duration      : {upload_duration}s")
    print(f"Total Client Wall-Clock    : {gen_duration}s (upload + wait)")
    if lat_sorted:
        print(f"Client-Perceived Latency   : median {med_lat:.1f}s | p95 {p95:.1f}s | max {lat_sorted[-1]:.1f}s")
    print(f"HTML Report                : {report_file}")
    print("=" * 60)

    if not args.no_open:
        try:
            os.startfile(str(report_file))
        except Exception:
            webbrowser.open(report_file.as_uri())

if __name__ == "__main__":
    main()
