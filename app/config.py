import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator
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
    API_KEY: str = "dev-insecure-secret-key-change-me"
    CORS_ORIGINS: List[str] = ["*"]
    MAX_UPLOAD_MB: int = 20
    LOG_LEVEL: str = "INFO"

    # ---------------- HF & Weights ----------------
    HF_TOKEN: Optional[str] = Field(default=None, validation_alias="HF_TOKEN")
    HUGGINGFACE_TOKEN: Optional[str] = Field(default=None, validation_alias="HUGGINGFACE_TOKEN")

    # ---------------- Directories ----------------
    PROJECT_DIR: Path = BASE_DIR
    AVATAR_DIR: Path = BASE_DIR / "Avatar"
    CROP_DIR: Path = BASE_DIR / "crop"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    REJECT_DIR: Path = BASE_DIR / "rejected"
    VAL_TMP_DIR: Path = BASE_DIR / "output" / "_val_tmp"

    # Worker paths
    DEEPFACE_WORKER_PATH: Path = BASE_DIR / "app" / "deepface_worker.py"
    VAL_WORKER_PATH: Path = BASE_DIR / "app" / "val_worker.py"
    VAL_WORKER_LOG: Path = BASE_DIR / "val_worker.log"
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

    FALLBACK_SCALE: float = 1.15
    HIJAB_SCALE: float = 1.25
    REF_SIZE: int = 1024
    STYLE_MODE: str = "comic"  # "semi_real" | "comic" | "off"
    KEEP_USER_EXPRESSION: bool = True

    # ---------------- Visor & Similarity ----------------
    USE_VISOR_REF: bool = True
    VALIDATE: bool = True
    BEST_OF_N: int = 3
    VISOR_EXTRA_TRIES: int = 2
    TIME_BUDGET: float = 40.0
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

    @property
    def effective_hf_token(self) -> Optional[str]:
        return self.HF_TOKEN or self.HUGGINGFACE_TOKEN

settings = Settings()
