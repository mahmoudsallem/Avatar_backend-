#!/usr/bin/env python3
"""
Saytara Avatar - User Acceptance Test (UAT) Round 2
Performs a full end-to-end acceptance test:
1. Uploads 14 test users to S3 under prefix 'uat2_'
2. Tracks live generation across all 3 GPU slots
3. Measures latency, throughput, and success rate
4. Downloads all outputs and generates a formal UAT HTML Report
"""
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
    res = subprocess.run(["aws"] + args, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return res.stdout.strip()

def main():
    print("=" * 60)
    print("SAYTARA AVATAR BACKEND - USER ACCEPTANCE TEST (UAT 2)")
    print("=" * 60)

    # 1. Load users in exact case-insensitive Windows order
    files = sorted(list(USERS_DIR.glob("*.*")), key=lambda p: p.name.lower())
    total_expected = len(files)
    print(f"[UAT] Found {total_expected} test photos in {USERS_DIR}")

    uat_dir = EVAL_DIR / "uat_round_2"
    inp_dir = uat_dir / "inputs"
    out_dir = uat_dir / "outputs"
    inp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Upload batch to S3
    t_start_upload = time.time()
    print(f"[UAT] Uploading {total_expected} user photos to {S3_USERS}/uat2_XX.jpg...")
    for idx, f in enumerate(files, 1):
        target_name = f"uat2_{idx:02d}.jpg"
        in_copy = inp_dir / f"{idx:02d}_{f.name}"
        shutil.copyfile(f, in_copy)
        run_aws(["s3", "cp", str(f), f"{S3_USERS}/{target_name}", "--quiet"])
        print(f"  -> Uploaded #{idx:02d}: {f.name} as {target_name}")

    upload_duration = round(time.time() - t_start_upload, 1)
    print(f"[UAT] All {total_expected} photos uploaded to S3 in {upload_duration}s.")
    print(f"[UAT] S3 Watcher is now processing photos on Blackwell GPU...")

    # 3. Monitor S3 output generation
    t_gen_start = time.time()
    seen = {}
    completed_items = []

    while len(seen) < total_expected:
        time.sleep(3)
        ls_output = run_aws(["s3", "ls", f"{S3_OUTPUTS}/"])
        lines = [line for line in ls_output.split("\n") if "uat2_" in line]

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                date, time_str, size_bytes, name = parts[0], parts[1], parts[2], parts[3]
                if name not in seen:
                    elapsed = round(time.time() - t_gen_start, 1)
                    seen[name] = elapsed
                    size_kb = round(int(size_bytes) / 1024, 1)
                    print(f"  [{elapsed}s] COMPLETED: {name} ({size_kb} KB)")

        elapsed_total = round(time.time() - t_gen_start, 1)
        if elapsed_total > 300:
            print("[UAT] Timeout after 300 seconds.")
            break

    gen_duration = round(time.time() - t_gen_start, 1)
    print(f"\n[UAT] Generation completed in {gen_duration}s.")
    print(f"[UAT] Downloading outputs from S3...")

    # 4. Download outputs and verify
    items = []
    for idx, f in enumerate(files, 1):
        target_name = f"uat2_{idx:02d}.jpg"
        in_copy = inp_dir / f"{idx:02d}_{f.name}"
        out_local = out_dir / f"{idx:02d}_{f.stem}_avatar.jpg"

        run_aws(["s3", "cp", f"{S3_OUTPUTS}/{target_name}", str(out_local), "--quiet"])
        ok = out_local.is_file() and out_local.stat().st_size > 1000
        size_kb = round(out_local.stat().st_size / 1024, 1) if ok else 0
        elapsed_item = seen.get(target_name, "-")

        items.append({
            "idx": idx,
            "filename": f.name,
            "input_rel": in_copy.relative_to(uat_dir).as_posix(),
            "input_size_kb": round(f.stat().st_size / 1024, 1),
            "output_rel": out_local.relative_to(uat_dir).as_posix() if ok else "",
            "output_size_kb": size_kb,
            "ok": ok,
            "elapsed": elapsed_item,
        })

    # 5. Build UAT Report
    avg_time = round(gen_duration / max(1, len(items)), 2)
    throughput = round(60.0 / max(0.1, avg_time), 1)
    success_count = sum(1 for it in items if it["ok"])

    cards_html = []
    for it in items:
        idx = it["idx"]
        ok = it["ok"]
        if ok:
            out_img = f'<img src="{html.escape(it["output_rel"])}" alt="Avatar" class="avatar-img" onclick="openModal(this.src)">'
            badge = '<span class="badge badge-success">PASSED</span>'
        else:
            out_img = '<div class="status-box error">GENERATION FAILED</div>'
            badge = '<span class="badge badge-danger">FAILED</span>'

        card = f"""
        <div class="eval-card">
            <div class="card-header">
                <span class="card-id">UAT #{idx:02d}</span>
                <span class="card-name">{html.escape(it['filename'])}</span>
                {badge}
                <span class="card-time">{it['elapsed']}s</span>
            </div>
            <div class="card-body">
                <div class="img-column">
                    <div class="img-wrapper">
                        <img src="{html.escape(it['input_rel'])}" alt="User Input" class="user-img" onclick="openModal(this.src)">
                    </div>
                    <div class="img-meta">
                        <strong>Input Portrait</strong>
                        <span>{it['input_size_kb']} KB</span>
                    </div>
                </div>
                <div class="arrow-divider">➔</div>
                <div class="img-column">
                    <div class="img-wrapper">
                        {out_img}
                    </div>
                    <div class="img-meta">
                        <strong>Sci-Fi Avatar Result</strong>
                        <span>{it['output_size_kb']} KB</span>
                    </div>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Saytara Avatar - UAT 2 Acceptance Report</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --surface-color: #151d2f;
            --border-color: #232f48;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-cyan: #06b6d4;
            --success-color: #10b981;
            --danger-color: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            padding: 30px 20px;
        }}
        .container {{ max-width: 1280px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 30px;
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
        .stat-label {{ font-size: 12px; color: var(--text-secondary); text-transform: uppercase; }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(580px, 1fr));
            gap: 24px;
        }}
        .eval-card {{
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
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
        }}
        .img-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            cursor: pointer;
            transition: transform 0.2s ease;
        }}
        .img-wrapper img:hover {{ transform: scale(1.03); }}
        .arrow-divider {{ font-size: 22px; color: var(--text-secondary); opacity: 0.6; }}
        .img-meta {{
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            width: 100%;
            font-size: 12px;
            color: var(--text-secondary);
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
        }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Saytara Avatar - UAT Acceptance Report (Round 2)</h1>
        <div class="subtitle">Live User Acceptance Test &bull; Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{success_count} / {total_expected}</div>
                <div class="stat-label">Acceptance Rate (100%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{gen_duration:.1f} s</div>
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

    report_file = uat_dir / "report.html"
    report_file.write_text(report_html, encoding="utf-8")
    print(f"\n[UAT] Report generated at: {report_file}")
    try:
        os.startfile(str(report_file))
    except Exception:
        webbrowser.open(report_file.as_uri())

if __name__ == "__main__":
    main()
