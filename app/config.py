import os
from pathlib import Path
from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------- Server & Security ----------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    LOG_LEVEL: str = "INFO"

    # ---------------- Output Encoding ----------------
    OUTPUT_MAX_BYTES: int = 300 * 1024  # final avatar JPEG must not exceed this

    # ---------------- Directories ----------------
    PROJECT_DIR: Path = BASE_DIR
    AVATAR_DIR: Path = BASE_DIR / "Avatar"
    CROP_DIR: Path = BASE_DIR / "crop"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    REJECT_DIR: Path = BASE_DIR / "rejected"
    VAL_TMP_DIR: Path = BASE_DIR / "output" / "_val_tmp"

    # Worker path - one script, dispatched by a "detect" / "embed" mode argument
    WORKER_PATH: Path = BASE_DIR / "app" / "deepface_worker.py"
    WORKER_LOG: Path = BASE_DIR / "worker.log"
    WORKER_TIMEOUT: int = 600

    # ---------------- Models ----------------
    CLIP_DIR: Path = Path("/Models/clip-vit-base-patch32")
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    FLUX_MODEL: str = "black-forest-labs/FLUX.2-klein-9B"
    LORA: str = "sisniha/BFS-Best-Face-Swap"
    LORA_FILE: str = "bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors"

    # ---------------- Face Detection Rules ----------------
    FACE_DETECTOR: str = "retinaface"
    MIN_FACE_CONFIDENCE: float = 0.90
    MIN_FACE_SIZE_PX: int = 80
    MAIN_FACE_DOMINANCE: float = 2.5
    CROP_SCALE: float = 2.0

    # ---------------- Step 1 Classifiers ----------------
    SCOPE_MIN_PERSON_SCORE: float = 0.35
    MIN_GENDER_CONFIDENCE: float = 75.0
    MIN_CLIP_GENDER_CONFIDENCE: float = 0.75
    GLASSES_POSITIVE_THRESHOLD: float = 0.58
    GLASSES_MIN_SCORE_MARGIN: float = 0.08
    BEARD_POSITIVE_THRESHOLD: float = 0.56
    BEARD_MIN_SCORE_MARGIN: float = 0.05
    HIJAB_POSITIVE_THRESHOLD: float = 0.50
    HIJAB_MIN_SCORE_MARGIN: float = 0.00

    # Bald/hairless men (male avatar only). Needs BOTH a scalp-colour check and CLIP to agree.
    DETECT_BALD: bool = True
    BALD_POSITIVE_THRESHOLD: float = 0.52
    BALD_MIN_SCORE_MARGIN: float = 0.03
    BALD_FALLBACK_CLIP: float = 0.70  # CLIP-only confidence needed when the scalp test cannot judge (scalp outside the crop, or the strip above the face is just background). Do NOT lower: a false "bald" deletes a real hair style.
    BALD_STRONG_CLIP: float = 0.85    # CLIP alone is trusted over a "not bare" scalp-colour test only at this very high confidence
    # Output guard: after each generated candidate of a BALD user, check that the avatar did not grow hair.
    VERIFY_BALD_OUTPUT: bool = True
    OUT_BALD_MIN: float = 0.50        # CLIP "bald" share of the avatar head needed to count the candidate as bald
    W_HAIR: float = 5.0               # score penalty for a candidate that has hair although the user is bald
    HAIR_EXTRA_TRIES: int = 2         # extra generations (beyond BEST_OF_N) when every candidate still has hair

    # ---------------- GPU Concurrency ----------------
    # Number of avatars generated at the same time on the GPU. Each extra slot loads another copy of the
    # FLUX transformer (~18 GB bf16); the text encoder + VAE are shared between slots.
    GPU_SLOTS: int = 2               # 0 = AUTO: keep adding slots while at least SLOT_MIN_FREE_GB VRAM stays free (max 8)
    SLOT_MIN_FREE_GB: float = 44.0   # don't load another slot unless this much VRAM is still free (each running job peaks at ~19 GB of activations)
    ANALYSIS_CONCURRENCY: int = 4    # Step 1 (CPU subprocesses) running at once, outside the GPU slots

    # ---------------- Speed options (all OFF by default = identical behaviour to before) ----------------
    # Exact / near-exact: they must not change the picture (check with bench_speed.py before enabling).
    CACHE_PROMPT_EMBEDS: bool = False  # cache the text-encoder output per distinct prompt (bit-identical)
    CACHE_REF_LATENTS: bool = False    # cache VAE latents of repeated reference images, e.g. the templates (bit-identical)
    FUSE_LORA: bool = False            # merge the LoRA into the weights; needs a uniform ID_LORA_SCHEDULE (rounding-level change)
    ATTENTION_BACKEND: str = ""        # "" = diffusers default; try "_native_cudnn" or "native" (rounding-level change)
    COMPILE_TRANSFORMER: bool = False  # torch.compile the FLUX transformer (use together with FUSE_LORA; slow first start)
    COMPILE_MODE: str = "default"      # "default" | "max-autotune-no-cudagraphs"
    WARMUP_AT_STARTUP: bool = True     # with COMPILE_TRANSFORMER: compile every template shape before serving
    CUDNN_ENABLED: bool = False        # the old code forced cuDNN off; True may speed up the VAE convolutions
    PERSISTENT_DETECT: bool = False    # keep DeepFace face-detection processes alive instead of spawning one per photo
    DETECT_WORKERS: int = 2            # how many persistent detection processes (each ~2 GB RAM)

    # ---------------- Generation Settings ----------------
    STEPS: int = 20
    CFG: float = 3.0
    REFINE_CFG: float = 2.0
    LORA_STRENGTH: float = 1.1
    ID_LORA_MULT: float = 1.0
    SEED: int = 42
    HEAD_CROP_SCALE: float = 1.6
    REFINE_IDENTITY: bool = True
    REFINE_STRENGTH: float = 0.85
    FACE_CROP_SCALE: float = 0.75
    CLEAN_BG: bool = True
    LORA_SELFTEST: bool = True  # at startup, prove the BFS LoRA really changes the output; auto-pick how to apply its strength
    DEBUG_DUMP: bool = False  # save face ref / visor ref / prompt / every candidate to output/debug/<key>/
    VAE_TILING: bool = False  # False = same as notebook (no tiled VAE decode)

    FALLBACK_SCALE: float = 1.15
    HIJAB_SCALE: float = 1.25
    REF_SIZE: int = 1024
    STYLE_MODE: str = "comic"  # "semi_real" | "comic" | "off"
    PROMPT_VARIANT: str = "tuned"  # "tuned" = the notebook prompt (default) | "id_focus" = identity-first prompt (A/B test with bench_prompt_ab.py before switching)
    KEEP_USER_EXPRESSION: bool = True

    # ---------------- Visor & Similarity ----------------
    USE_VISOR_REF: bool = True
    VALIDATE: bool = True
    BEST_OF_N: int = 3
    VISOR_EXTRA_TRIES: int = 2
    TIME_BUDGET: float = 0.0  # 0 = no limit (notebook behaviour)
    ID_LORA_SCHEDULE: List[float] = [1.0, 1.1, 1.2]
    STOP_ID: float = 0.55
    W_JAW: float = 0.8
    W_NO_VISOR: float = 5.0
    VERIFY_MODEL: str = "ArcFace"
    VERIFY_DETECTOR: str = "retinaface"
    VERIFY_MAX_SIDE: int = 512

    # Glasses Validation
    VALIDATE_GLASSES: bool = True
    VISOR_MIN_SCORE: float = 0.38
    DARK_FRAME_RATIO: float = 1.3
    MAX_GLASSES_FIXES: int = 3
    ADD_AVATAR_GLASSES: bool = True
    USER_GLASSES_MODE: str = "avatar"

    @field_validator("CLIP_MODEL", mode="before")
    @classmethod
    def resolve_clip_model(cls, v: Optional[str]) -> str:
        if v:
            return v
        local_clip = Path("/Models/clip-vit-base-patch32")
        if local_clip.exists():
            return str(local_clip)
        return "openai/clip-vit-base-patch32"

settings = Settings()
