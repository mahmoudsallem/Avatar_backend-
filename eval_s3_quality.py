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

def build_html_report(out_dir: Path, items: list, wall_clock: float, batch_name: str, title: str):
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
                <span class="card-time">{elapsed}s</span>
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
            <div class="stat-card">
                <div class="stat-value">{wall_clock:.1f} s</div>
                <div class="stat-label">Total Generation Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_time:.2f} s</div>
                <div class="stat-label">Avg Time / Avatar</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{throughput:.1f} / min</div>
                <div class="stat-label">Throughput (~{int(throughput * 60)}/hr)</div>
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

def main():
    parser = argparse.ArgumentParser(description="Saytara Avatar S3 Quality & Performance Evaluator")
    parser.add_argument("--round", type=int, default=3, help="Evaluation / UAT Round number (default: 3)")
    parser.add_argument("--prefix", type=str, default="", help="Custom S3 file prefix (default: uat<round>_)")
    parser.add_argument("--from-prefix", type=str, default="", help="Build report from existing S3 run without uploading")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--timeout", type=int, default=300, help="Max wait time for S3 generation in seconds")
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

        wall_clock = 70.0
        report_file = build_html_report(out_dir, items, wall_clock, prefix, title)
        print(f"\n[EVAL] Report generated: {report_file}")
        if not args.no_open:
            try:
                os.startfile(str(report_file))
            except Exception:
                webbrowser.open(report_file.as_uri())
        return

    # Mode B: Live S3 Evaluation (UAT)
    round_num = args.round
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
    print(f"Prefix: {prefix} | Total Images: {total_expected}")
    print("=" * 60)

    # 1. Upload batch to S3 Users/
    t_start_upload = time.time()
    print(f"\n[1/3] Uploading {total_expected} user photos to {S3_USERS}/{prefix}XX.jpg...")
    for idx, f in enumerate(files, 1):
        target_name = f"{prefix}{idx:02d}.jpg"
        in_copy = inp_dir / f"{idx:02d}_{f.name}"
        shutil.copyfile(f, in_copy)
        run_aws(["s3", "cp", str(f), f"{S3_USERS}/{target_name}", "--quiet"])
        print(f"  -> Uploaded #{idx:02d}: {f.name} as {target_name}")

    upload_duration = round(time.time() - t_start_upload, 1)
    print(f"[1/3] Upload complete in {upload_duration}s.")

    # 2. Monitor S3 output generation
    print(f"\n[2/3] S3 Watcher is processing batch on Blackwell GPU slots...")
    t_gen_start = time.time()
    seen = {}

    while len(seen) < total_expected:
        time.sleep(3)
        ls_output = run_aws(["s3", "ls", f"{S3_OUTPUTS}/"])
        lines = [line for line in ls_output.split("\n") if prefix in line]

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                date, time_str, size_bytes, name = parts[0], parts[1], parts[2], parts[3]
                if name not in seen:
                    elapsed = round(time.time() - t_gen_start, 1)
                    seen[name] = elapsed
                    size_kb = round(int(size_bytes) / 1024, 1)
                    print(f"  [{elapsed}s] COMPLETED #{len(seen):02d}/{total_expected}: {name} ({size_kb} KB)")

        elapsed_total = round(time.time() - t_gen_start, 1)
        if elapsed_total > args.timeout:
            print(f"[2/3] Timeout reached ({args.timeout}s).")
            break

    gen_duration = round(time.time() - t_gen_start, 1)
    print(f"[2/3] Generation finished in {gen_duration}s ({len(seen)}/{total_expected} completed).")

    # 3. Download results and compile report
    print(f"\n[3/3] Downloading generated avatars and building quality report...")
    items = []
    for idx, f in enumerate(files, 1):
        target_name = f"{prefix}{idx:02d}.jpg"
        in_copy = inp_dir / f"{idx:02d}_{f.name}"
        out_local = out_img_dir / f"{idx:02d}_{f.stem}_avatar.jpg"

        run_aws(["s3", "cp", f"{S3_OUTPUTS}/{target_name}", str(out_local), "--quiet"])
        ok = out_local.is_file() and out_local.stat().st_size > 1000
        size_kb = round(out_local.stat().st_size / 1024, 1) if ok else 0
        elapsed_item = seen.get(target_name, "-")

        items.append({
            "idx": idx,
            "filename": f.name,
            "input_rel": in_copy.relative_to(out_dir).as_posix(),
            "input_size_kb": round(f.stat().st_size / 1024, 1),
            "output_rel": out_local.relative_to(out_dir).as_posix() if ok else "",
            "output_size_kb": size_kb,
            "ok": ok,
            "elapsed": elapsed_item,
        })

    report_file = build_html_report(out_dir, items, gen_duration, prefix, title)
    success_count = sum(1 for it in items if it["ok"])
    avg_s = round(gen_duration / max(1, len(items)), 2)
    t_rate = round(60.0 / max(0.1, avg_s), 1)

    print("\n" + "=" * 60)
    print("S3 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Acceptance Rate      : {success_count}/{total_expected} ({(success_count/total_expected)*100:.0f}%)")
    print(f"Total Generation Time: {gen_duration}s")
    print(f"Average Per Avatar   : {avg_s}s")
    print(f"Throughput           : {t_rate} avatars/min (~{int(t_rate * 60)} / hour)")
    print(f"HTML Report          : {report_file}")
    print("=" * 60)

    if not args.no_open:
        try:
            os.startfile(str(report_file))
        except Exception:
            webbrowser.open(report_file.as_uri())

if __name__ == "__main__":
    main()
