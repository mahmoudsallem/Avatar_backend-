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
    # TEST ONLY: lets /v1/avatar accept ?steps=&tries=&time_budget=&cfg=&style= per request so settings can be A/B tested
    # on the same photos (send_10_images.py --matrix). Keep false in production.
    ALLOW_TEST_OVERRIDES: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", 8080))
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

    # ---------------- S3 Folder Watcher ----------------
    # Polls s3://S3_BUCKET/S3_USERS_PREFIX for new photos, generates an avatar for
    # each, and uploads it to s3://S3_BUCKET/S3_GENERATED_PREFIX under the same
    # base filename (always as .jpg - the pipeline's output is always JPEG).
    # AWS credentials come from the normal boto3 chain (env vars, ~/.aws/credentials,
    # or an EC2 instance role) - nothing extra is needed here.
    S3_ENABLED: bool = False
    S3_BUCKET: str = "avatar-etisalata"
    S3_USERS_PREFIX: str = "Users/"
    S3_GENERATED_PREFIX: str = "Generated_Avatar/"
    S3_REGION: str = "us-east-1"
    S3_POLL_INTERVAL: float = 5.0     # seconds between folder listings (fallback mode, only used when S3_QUEUE_URL is empty)
    S3_MAX_CONCURRENT: int = 2        # user photos the watcher processes at once

    # Real-time mode: an SQS queue fed by an S3 "ObjectCreated" event notification on Users/*.
    # When set, the watcher long-polls this queue instead of periodically listing Users/ - a new
    # photo is picked up within ~1-2s instead of waiting up to S3_POLL_INTERVAL. Leave empty to
    # fall back to the polling loop (e.g. local/dev runs with no SQS set up).
    S3_QUEUE_URL: str = ""
    S3_QUEUE_WAIT_SECONDS: int = 20   # SQS long-poll wait (max 20); keeps the loop from hammering SQS with empty receives

    # Model cache (install_models.py): a separate bucket (not the photo bucket above) that the
    # HF hub cache / DeepFace weights / local CLIP dir get mirrored to/from.
    S3_MODEL_BUCKET: str = "saytara-models"
    S3_MODEL_CACHE_PREFIX: str = ""

    # ---------------- Models ----------------
    CLIP_DIR: Path = Path("/Models/clip-vit-base-patch32")
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    FLUX_MODEL: str = "black-forest-labs/FLUX.2-klein-9B"
    FLUX_BACKEND: str = "diffusers"  # "diffusers" or "nunchaku"
    NUNCHAKU_MODEL: str = "tonera/FLUX.2-klein-9B-Nunchaku"
    LORA: str = "sisniha/BFS-Best-Face-Swap"
    LORA_FILE: str = "bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors"
    # Trigger word put at the START of every prompt (empty = none). The SmartCharacterSwap LoRA needs "jhuangswap".
    LORA_TRIGGER: str = ""

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
    BALD_SCALP_CLIP: float = 0.65     # CLIP confidence needed IN ADDITION to a "bare" scalp-colour test (a tall/receding forehead fools the colour test)
    BALD_FALLBACK_CLIP: float = 0.80  # CLIP-only confidence needed when the scalp test cannot judge (scalp outside the crop, or the strip above the face is just background). Do NOT lower: a false "bald" deletes a real hair style.
    BALD_STRONG_CLIP: float = 0.90    # CLIP alone is trusted over a "not bare" scalp-colour test only at this very high confidence
    # Reverse output guard: user HAS hair but the avatar came out bald -> penalise + retry with HAIR_RETRY.
    VERIFY_HAIR_OUTPUT: bool = True
    OUT_HAIR_BALD_MAX: float = 0.80   # CLIP "bald" share of the avatar head above which a candidate of a user WITH hair counts as bald
    # Output guard: after each generated candidate of a BALD user, check that the avatar did not grow hair.
    VERIFY_BALD_OUTPUT: bool = True
    OUT_BALD_MIN: float = 0.50        # CLIP "bald" share of the avatar head needed to count the candidate as bald
    W_HAIR: float = 5.0               # score penalty for a candidate that has hair although the user is bald
    HAIR_EXTRA_TRIES: int = 2         # extra generations (beyond BEST_OF_N) when every candidate still has hair

    # Hair-VOLUME output guard (men with hair, not bald): the fast STEPS=4/CFG=1.0 profile sometimes drifts hair
    # volume/texture away from the user's real photo (e.g. a buzz cut comes out fuller and swept back) even though
    # the HAIR_USER prompt already says not to. This re-checks volume per candidate and retries a few times instead
    # of slowing down every generation to fix it.
    VERIFY_HAIR_VOLUME: bool = True
    HAIR_VOLUME_MISMATCH: float = 0.35   # |candidate short-hair score - user short-hair score| at/above this = mismatch
    W_HAIR_VOLUME: float = 2.0           # score penalty for a hair-volume mismatch
    HAIR_VOLUME_EXTRA_TRIES: int = 1     # extra generations (beyond BEST_OF_N) when every candidate still mismatches

    # ---------------- GPU Concurrency ----------------
    # Number of avatars generated at the same time on the GPU. Each extra slot loads another copy of the
    # FLUX transformer (~18 GB bf16); the text encoder + VAE are shared between slots.
    SLOT_PEAK_GB: float = 19.0       # activation memory one running job can peak at; after loading, slots are trimmed until free VRAM >= slots x this (OOM guard)
    OOM_RETRY: bool = True           # on a CUDA out-of-memory error during generation: free the cache, wait, retry that candidate once
    VAL_WORKERS: int = 1             # persistent DeepFace/ArcFace scoring processes (CPU). 1 = old behaviour; with GPU_SLOTS=3 use 3 so slots do not queue for scoring
    SCHED_SHIFT: float = 1.0         # FlowMatch scheduler shift. 1.0 = unchanged; 1.5-2.0 spends more of the few steps on face structure (identity)
    VAE_COMPILE: bool = False        # torch.compile the VAE decoder (falls back to the normal decode if compile fails)
    GPU_SLOTS: int = 2               # 0 = AUTO: keep adding slots while at least SLOT_MIN_FREE_GB VRAM stays free (max 8)
    SLOT_MIN_FREE_GB: float = 44.0   # don't load another slot unless this much VRAM is still free (each running job peaks at ~19 GB of activations)
    ANALYSIS_CONCURRENCY: int = 4    # Step 1 (CPU subprocesses) running at once, outside the GPU slots

    # ---------------- Speed options (all OFF by default = identical behaviour to before) ----------------
    # Exact / near-exact: they must not change the picture (check with bench_speed.py before enabling).
    CACHE_PROMPT_EMBEDS: bool = False  # cache the text-encoder output per distinct prompt (bit-identical)
    CACHE_REF_LATENTS: bool = False    # cache VAE latents of repeated reference images, e.g. the templates (bit-identical)
    FUSE_LORA: bool = False            # merge the LoRA into the weights; needs a uniform ID_LORA_SCHEDULE (rounding-level change)
    ATTENTION_BACKEND: str = ""        # diffusers backend only (ignored by nunchaku). "" = PyTorch SDPA default. sm_120 (RTX PRO 6000 / g7e): use "_native_cudnn", "_native_flash" or "native". "flash4" needs sm90/sm100 and "flash3" needs sm90 - both are skipped with a warning on sm_120. Exact name "flash_4_hub" bypasses the gate
    COMPILE_TRANSFORMER: bool = False  # torch.compile the FLUX transformer (use together with FUSE_LORA; slow first start)
    COMPILE_MODE: str = "default"      # "default" | "max-autotune-no-cudagraphs" (inductor only)
    COMPILE_BACKEND: str = "inductor"  # "inductor" | "tensorrt" (torch-tensorrt; diffusers backend + FUSE_LORA only, NOT with nunchaku)
    MODEL_DTYPE: str = "bf16"          # diffusers backend: "bf16" (default) | "fp16" - dtype of the FLUX pipeline weights (no quantization either way)
    LOAD_DIRECT_TO_GPU: bool = True   # diffusers backend: load the FLUX weights straight onto the GPU (device_map="cuda") instead of into CPU RAM first, then .to("cuda")
    TRT_ENGINE_CACHE_DIR: str = "trt_cache"  # TensorRT engines are saved here and reused on the next start (first start builds them: slow)
    TRT_PRECISION: str = "bf16"        # COMPILE_BACKEND=tensorrt only: "bf16" (default) | "fp16" (transformer cast to fp16; checked for NaN/black output at warm-up)
    WARMUP_AT_STARTUP: bool = True     # with COMPILE_TRANSFORMER: compile every template shape before serving
    WARMUP_RUNTIME: bool = True        # at startup run a throw-away generation on every GPU slot + template, and wake every CPU worker, so the first real batch is not slow (works with any backend)
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
    # Skin-only clean-up (light smoothing, shine removal, tone nudged toward the user's real skin) for these avatars.
    # No hijab recolour / drape painting - that is HIJAB_POSTFIX, kept off.
    # Off: the smoothing / shine compression made women's skin flat and hazy next to the crisp male avatars.
    SKIN_POSTFIX_AVATARS: List[str] = []
    SKIN_POSTFIX_SMOOTH: float = 0.35
    SKIN_POSTFIX_MATCH: float = 0.6
    HIJAB_POSTFIX: bool = True  # pixel post-process for hijab avatars (skin smooth/tone match, neck-cowl recolour)
    HIJAB_SKIN_SMOOTH: float = 0.8
    HIJAB_SKIN_MATCH: float = 0.6
    HIJAB_COLOR_MATCH: float = 0.85
    HIJAB_DRAPE_FIX: float = 0.9
    REF_SIZE: int = 1024
    STYLE_MODE: str = "comic"  # "semi_real" | "comic" | "off"
    PROMPT_VARIANT: str = "tuned"  # "tuned" = the notebook prompt (default) | "id_focus" = identity-first prompt | "v3" = short positive-only prompt (~380 words vs ~1300), A/B with send_10_images.py | "v4" = v3 + per-photo measured face traits (face shape, skin tone, beard style, hair type) with a stronger identity block
    KEEP_USER_EXPRESSION: bool = True

    # ---------------- Visor & Similarity ----------------
    USE_VISOR_REF: bool = True
    # Picture 3 cleanup: replace the template close-up (template eyes/brows/nose behind a dark-blue lens) with the
    # traced visor alone on black + a clear see-through lens. Stops template eyes / blue irises / oversized visor.
    VISOR_REF_CLEAN: List[str] = ["Woman", "Woman_Hijab", "Man"]
    # Remove the blue lens tint from the TEMPLATE (Picture 1 = base image) for these avatars: its tinted lens made the
    # user's eyes come out blue/purple. Only the lens area inside the traced visor outline is touched.
    TEMPLATE_LENS_DETINT: List[str] = ["Woman", "Woman_Hijab", "Man"]
    # OPT-IN (A/B first): cut the far-side part of the visor that sticks out past the cheek, in the template (inpainted)
    # and in the visor reference. e.g. VISOR_TRIM_FAR_SIDE=["Woman","Woman_Hijab"]
    VISOR_TRIM_FAR_SIDE: List[str] = []
    # Women/hijab prompt tune (id_focus/tuned): user's own eyes block, smooth clean skin instead of "keep wrinkles/pores",
    # visor fitted to her face. false = exact previous prompt.
    WOMEN_PROMPT_TUNE: bool = True
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
