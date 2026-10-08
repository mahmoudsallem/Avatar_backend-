"""Benchmark + accuracy check for the speed options (run ON THE GPU SERVER, with the app STOPPED).

Each configuration loads the models once (1 GPU slot, TIME_BUDGET=0, fixed seeds), generates the same photos
and saves the avatars. `compare` then reports, against the right baseline:
    speed       seconds per avatar (Step 1 / generation / total) and the speed-up
    accuracy    pixel difference of the outputs (mean abs 0-255, PSNR) and the ArcFace identity score

    python bench_speed.py suite                      # run every configuration below, then compare
    python bench_speed.py suite --only baseline,cache,fuse_compile
    python bench_speed.py run --name mytest --env CACHE_PROMPT_EMBEDS=true --env ATTENTION_BACKEND=native
    python bench_speed.py compare                    # re-print the table from saved results

Outputs go to bench_results/<name>/ (avatars + results.json) and bench_results/summary.md.
Read it like this: "mean abs diff" ~0 and PSNR > 45 dB = visually identical; identical best_try and an id_sim that
moves by less than ~0.01 = same accuracy. A change that wins on speed but shifts id_sim or picks other tries is NOT lossless.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "bench_results"
UNIFORM = "[1.0,1.0]"   # FUSE_LORA needs one fixed LoRA strength for all tries

# name, compare-against, env overrides
CONFIGS = [
    ("baseline",         None,               {}),
    ("cache",            "baseline",         {"CACHE_PROMPT_EMBEDS": "true", "CACHE_REF_LATENTS": "true"}),
    ("detect",           "baseline",         {"PERSISTENT_DETECT": "true"}),
    ("attn_native",      "baseline",         {"ATTENTION_BACKEND": "native"}),
    ("attn_cudnn",       "baseline",         {"ATTENTION_BACKEND": "_native_cudnn"}),
    ("cudnn_on",         "baseline",         {"CUDNN_ENABLED": "true"}),
    # --- these change the result on purpose: the table tells you how much (judge by the images + id_sim) ---
    ("steps_4",          "baseline",         {"STEPS": "4"}),
    ("steps_6",          "baseline",         {"STEPS": "6"}),
    ("ref_768",          "baseline",         {"REF_SIZE": "768"}),
    ("baseline_uniform", None,               {"ID_LORA_SCHEDULE": UNIFORM}),
    ("fuse",             "baseline_uniform", {"ID_LORA_SCHEDULE": UNIFORM, "FUSE_LORA": "true"}),
    ("fuse_compile",     "baseline_uniform", {"ID_LORA_SCHEDULE": UNIFORM, "FUSE_LORA": "true", "COMPILE_TRANSFORMER": "true"}),
    ("all_in",           "baseline_uniform", {"ID_LORA_SCHEDULE": UNIFORM, "FUSE_LORA": "true", "COMPILE_TRANSFORMER": "true",
                                              "CACHE_PROMPT_EMBEDS": "true", "CACHE_REF_LATENTS": "true",
                                              "PERSISTENT_DETECT": "true"}),
]
REF_OF = {n: r for n, r, _ in CONFIGS}


# --------------------------------------------------------------------------------------------- run one config
def run_config(a):
    for kv in a.env or []:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    os.environ.setdefault("GPU_SLOTS", "1")
    os.environ["GPU_SLOTS"] = "1"          # one slot: we measure single-avatar speed, not concurrency
    os.environ["TIME_BUDGET"] = "0"        # time-based early stops would make runs incomparable
    sys.path.insert(0, str(ROOT))
    from app import pipeline
    from app.config import settings

    photos = sorted(p for p in Path(a.photos).iterdir() if p.suffix.lower() in pipeline.IMAGE_EXTENSIONS)
    out_dir = OUT / a.name
    out_dir.mkdir(parents=True, exist_ok=True)

    t = time.time()
    pipeline.load_models()
    load_s = round(time.time() - t, 1)
    slot = pipeline.slots[0]
    print(f"[{a.name}] models loaded in {load_s}s | fused={slot.fused} | env={a.env}", flush=True)

    # analyse once per photo (Step 1 timed on its 2nd run so persistent workers / caches are warm)
    infos, rows = {}, []
    for ph in photos:
        try:
            t = time.time(); info = pipeline.analyse_user(ph); first = time.time() - t
            t = time.time(); info = pipeline.analyse_user(ph); step1 = time.time() - t
            infos[ph.name] = (info, round(step1, 2), round(first, 2))
        except Exception as e:
            print(f"[{a.name}] skip {ph.name}: {e}", flush=True)
        if len(infos) >= a.limit:
            break
    if not infos:
        sys.exit("no usable photos")

    first_name = next(iter(infos))
    pipeline.generate_avatar(Path(a.photos) / first_name, infos[first_name][0], None, slot)  # warm-up, not counted
    print(f"[{a.name}] warm-up done", flush=True)

    for name, (info, step1, step1_cold) in infos.items():
        totals, gens, vals, last = [], [], [], None
        for rep in range(a.repeat):
            t = time.time()
            jpeg, rec = pipeline.generate_avatar(Path(a.photos) / name, info, None, slot)
            totals.append(time.time() - t); gens.append(rec["gen_s"]); vals.append(rec["val_s"])
            if rep == 0:
                (out_dir / (Path(name).stem + ".jpg")).write_bytes(jpeg)
                last = rec
        rows.append({
            "photo": name, "avatar": info["avatar"], "step1_s": step1, "step1_cold_s": step1_cold,
            "gen_s": round(statistics.mean(gens), 2), "val_s": round(statistics.mean(vals), 2),
            "total_s": round(statistics.mean(totals), 2),
            "tries": last["tries"], "best_try": last["best_try"], "id_sim": last["id_sim"],
            "visor": last["visor"], "score": last["score"], "output": Path(name).stem + ".jpg",
        })
        print(f"[{a.name}] {name}: step1 {step1}s | gen {rows[-1]['gen_s']}s | total {rows[-1]['total_s']}s | "
              f"tries {last['tries']} best {last['best_try']} id_sim {last['id_sim']}", flush=True)

    res = {"name": a.name, "env": a.env, "load_s": load_s, "fused": slot.fused,
           "cache_stats": dict(pipeline._CACHE_STATS), "rows": rows,
           "settings": {"STEPS": settings.STEPS, "CFG": settings.CFG, "BEST_OF_N": settings.BEST_OF_N}}
    (out_dir / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pipeline.shutdown_models()
    print(f"[{a.name}] saved {out_dir}", flush=True)


# --------------------------------------------------------------------------------------------- compare
def load(name):
    p = OUT / name / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def diff_images(a_path, b_path):
    import numpy as np
    from PIL import Image
    A = np.asarray(Image.open(a_path).convert("RGB")).astype("float32")
    B = np.asarray(Image.open(b_path).convert("RGB")).astype("float32")
    if A.shape != B.shape:
        return None
    d = np.abs(A - B)
    mse = float(((A - B) ** 2).mean())
    psnr = 99.0 if mse == 0 else 10 * __import__("math").log10(255 ** 2 / mse)
    return float(d.mean()), float(d.max()), psnr


def avg(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.mean(v) if v else float("nan")


def compare(_a=None):
    lines = []
    header = (f"{'config':<18}{'vs':<18}{'step1 s':>8}{'gen s':>7}{'total s':>8}{'speed-up':>9}"
              f"{'mean|diff|':>11}{'min PSNR':>9}{'id_sim':>8}{'d id_sim':>9}{'same try':>9}  verdict")
    lines.append(header); lines.append("-" * len(header))
    for name, ref, _ in CONFIGS:
        r = load(name)
        if r is None:
            continue
        rows = r["rows"]
        if ref is None:
            lines.append(f"{name:<18}{'(reference)':<18}{avg(rows,'step1_s'):>8.1f}{avg(rows,'gen_s'):>7.1f}"
                         f"{avg(rows,'total_s'):>8.1f}{'1.00x':>9}{'':>11}{'':>9}{avg(rows,'id_sim'):>8.3f}")
            continue
        b = load(ref)
        if b is None:
            lines.append(f"{name:<18}{ref:<18}  (reference '{ref}' not run yet)")
            continue
        brows = {x["photo"]: x for x in b["rows"]}
        diffs, psnrs, same_try, dsim = [], [], 0, []
        for x in rows:
            y = brows.get(x["photo"])
            if not y:
                continue
            d = diff_images(OUT / name / x["output"], OUT / ref / y["output"])
            if d:
                diffs.append(d[0]); psnrs.append(d[2])
            same_try += int(x["best_try"] == y["best_try"])
            if isinstance(x.get("id_sim"), (int, float)) and isinstance(y.get("id_sim"), (int, float)):
                dsim.append(x["id_sim"] - y["id_sim"])
        n = max(1, len(rows))
        speed = avg(b["rows"], "total_s") / avg(rows, "total_s")
        md = statistics.mean(diffs) if diffs else float("nan")
        mp = min(psnrs) if psnrs else float("nan")
        ds = statistics.mean(dsim) if dsim else float("nan")
        step_only = name == "detect"
        if step_only:
            verdict = "SAME (Step 1 only)" if abs(ds) < 0.01 else "CHECK"
        elif md < 1.0 and mp > 40 and same_try == n and abs(ds) < 0.01:
            verdict = "SAME"
        elif md < 3.0 and mp > 35 and same_try == n and abs(ds) < 0.02:
            verdict = "TINY (rounding) - look at the images"
        else:
            verdict = "DIFFERS - do not use"
        lines.append(f"{name:<18}{ref:<18}{avg(rows,'step1_s'):>8.1f}{avg(rows,'gen_s'):>7.1f}{avg(rows,'total_s'):>8.1f}"
                     f"{speed:>8.2f}x{md:>11.3f}{mp:>9.1f}{avg(rows,'id_sim'):>8.3f}{ds:>+9.3f}{same_try:>6}/{n:<2}  {verdict}")
    text = "\n".join(lines)
    print("\n" + text)
    (OUT / "summary.md").write_text("```\n" + text + "\n```\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------- suite
def suite(a):
    only = set(a.only.split(",")) if a.only else None
    if only:
        only |= {REF_OF[n] for n in list(only) if REF_OF.get(n)}      # a selected config always brings its baseline
    for name, ref, env in CONFIGS:
        if only and name not in only:
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--name", name, "--photos", a.photos,
               "--limit", str(a.limit), "--repeat", str(a.repeat)]
        for k, v in env.items():
            cmd += ["--env", f"{k}={v}"]
        print(f"\n=== {name} ===", flush=True)
        subprocess.run(cmd, check=False)          # one process per config = clean GPU memory and settings
    compare()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "suite"):
        p = sub.add_parser(cmd)
        p.add_argument("--photos", default="users")
        p.add_argument("--limit", type=int, default=4, help="how many photos (default 4)")
        p.add_argument("--repeat", type=int, default=2, help="timed repeats per photo (default 2)")
        if cmd == "run":
            p.add_argument("--name", required=True)
            p.add_argument("--env", action="append", help="KEY=VALUE override, repeatable")
        else:
            p.add_argument("--only", default=None, help="comma list of config names")
    sub.add_parser("compare")
    a = ap.parse_args()
    {"run": run_config, "suite": suite, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
