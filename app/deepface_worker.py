import os
import sys
import json
import warnings
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from deepface import DeepFace

def main():
    if len(sys.argv) < 7:
        print("__JSON__" + json.dumps({"error": "insufficient_arguments"}))
        return

    path, detector = sys.argv[1:3]
    scale = float(sys.argv[3])
    min_conf = float(sys.argv[4])
    min_size = int(sys.argv[5])
    dominance = float(sys.argv[6])

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
                x, y, w, h = (int(area[key]) for key in ("x", "y", "w", "h"))
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
                            "gender": {key: float(value) for key, value in analysis["gender"].items()},
                        }]
                    except Exception:
                        result["faces"] = [{
                            "facial_area": {"x": x, "y": y, "w": w, "h": h},
                            "confidence": float(face.get("confidence", 0)),
                            "gender": {},
                        }]
                result["background_faces_ignored"] = len(candidates) - 1

    print("__JSON__" + json.dumps(result))

if __name__ == "__main__":
    main()
