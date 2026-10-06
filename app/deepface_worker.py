"""Standalone CPU subprocess worker (TensorFlow/DeepFace never shares a process
with the PyTorch/FLUX side - mixing the two crashes on this stack).

Two modes, dispatched by argv[1]:
  detect <image> <detector> <crop_scale> <min_conf> <min_size> <dominance>
      One-shot: finds the single dominant face, rejects multi-face images,
      runs gender analysis, prints one __JSON__ line, exits. Spawned fresh
      per request via subprocess.run().
  embed <model> <detector>
      Persistent: prints __READY__, then reads {"img", "enforce"} JSON lines
      from stdin and emits one __JSON__ line of ArcFace embeddings per
      request until stdin closes. Started once at app startup and kept alive
      via subprocess.Popen().
"""
import os
import sys
import json
import warnings
import contextlib
import io

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import cv2

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from deepface import DeepFace

def run_detect(argv):
    if len(argv) < 6:
        print("__JSON__" + json.dumps({"error": "insufficient_arguments"}))
        return

    path, detector = argv[0], argv[1]
    scale = float(argv[2])
    min_conf = float(argv[3])
    min_size = int(argv[4])
    dominance = float(argv[5])

    image = cv2.imread(path)
    result = {"faces": [], "background_faces_ignored": 0}
    if image is None:
        result["error"] = "unreadable"
    else:
        height, width = image.shape[:2]
        try:
            detected = DeepFace.extract_faces(
                img_path=image,
                detector_backend=detector,
                enforce_detection=False,
                align=False,
            )
        except Exception:
            detected = []

        candidates = [
            face for face in detected
            if float(face.get("confidence", 0)) >= min_conf
            and int(face["facial_area"]["w"]) >= min_size
            and int(face["facial_area"]["h"]) >= min_size
        ]
        if candidates:
            candidates.sort(
                key=lambda face: face["facial_area"]["w"] * face["facial_area"]["h"],
                reverse=True,
            )
            largest_area = candidates[0]["facial_area"]["w"] * candidates[0]["facial_area"]["h"]
            main_faces = [
                face for face in candidates
                if face["facial_area"]["w"] * face["facial_area"]["h"] >= largest_area / dominance
            ]
            if len(main_faces) > 1:
                result["error"] = "multiple_main_faces"
                result["main_faces"] = len(main_faces)
            else:
                face = main_faces[0]
                area = face["facial_area"]
                x, y, w, h = (int(area[k]) for k in ("x", "y", "w", "h"))
                cx, cy = x + w // 2, y + h // 2
                nw, nh = int(w * scale), int(h * scale)
                crop = image[
                    max(0, cy - nh // 2):min(height, cy + nh // 2),
                    max(0, cx - nw // 2):min(width, cx + nw // 2),
                ]
                if crop.size:
                    try:
                        analysis = DeepFace.analyze(
                            img_path=crop,
                            actions=["gender"],
                            detector_backend="skip",
                            enforce_detection=False,
                            silent=True,
                        )
                        analysis = analysis[0] if isinstance(analysis, list) else analysis
                        result["faces"] = [{
                            "facial_area": {"x": x, "y": y, "w": w, "h": h},
                            "confidence": float(face.get("confidence", 0)),
                            "gender": {k: float(v) for k, v in analysis["gender"].items()},
                        }]
                    except Exception:
                        result["faces"] = [{
                            "facial_area": {"x": x, "y": y, "w": w, "h": h},
                            "confidence": float(face.get("confidence", 0)),
                            "gender": {},
                        }]
                result["background_faces_ignored"] = len(candidates) - 1

    print("__JSON__" + json.dumps(result))

def run_embed(argv):
    model = argv[0] if len(argv) > 0 else "ArcFace"
    detector = argv[1] if len(argv) > 1 else "retinaface"
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
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if mode == "detect":
        run_detect(rest)
    elif mode == "embed":
        run_embed(rest)
    else:
        print("__JSON__" + json.dumps({"error": f"unknown mode '{mode}', expected 'detect' or 'embed'"}))
