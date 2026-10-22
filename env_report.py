"""Print the library versions that matter. Run it in the NOTEBOOK environment and on the SERVER; the lines must match.
    python env_report.py"""
import importlib.metadata as md
for p in ("torch", "diffusers", "transformers", "tokenizers", "peft", "accelerate", "safetensors", "huggingface-hub", "numpy", "pillow"):
    try: print(f"{p}=={md.version(p)}")
    except Exception: print(f"{p}==NOT INSTALLED")
