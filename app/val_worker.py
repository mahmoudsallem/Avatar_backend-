import os
import sys
import json
import io
import contextlib
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from deepface import DeepFace

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "ArcFace"
    detector = sys.argv[2] if len(sys.argv) > 2 else "retinaface"
    print("__READY__", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            with contextlib.redirect_stdout(io.StringIO()):
                reps = DeepFace.represent(
                    img_path=req["img"],
                    model_name=model,
                    detector_backend=detector,
                    enforce_detection=bool(req["enforce"]),
                    align=True,
                )
            out = {
                "faces": [
                    {
                        "emb": [float(x) for x in r["embedding"]],
                        "w": int(r["facial_area"].get("w", 0)),
                        "h": int(r["facial_area"].get("h", 0)),
                    }
                    for r in reps
                ]
            }
        except Exception as e:
            out = {"error": type(e).__name__ + ": " + str(e)[:300]}
        print("__JSON__" + json.dumps(out), flush=True)

if __name__ == "__main__":
    main()
