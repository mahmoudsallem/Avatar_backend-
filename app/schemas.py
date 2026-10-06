from typing import List, Optional
from pydantic import BaseModel, Field

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message explaining failure reason")

class HealthResponse(BaseModel):
    status: str = Field("ok", description="Overall service status")
    cuda_available: bool = Field(..., description="Whether CUDA is available to PyTorch")
    device_name: Optional[str] = Field(None, description="Active CUDA device name")
    models_loaded: bool = Field(..., description="Whether CLIP and FLUX pipelines are fully loaded")
    clip_loaded: bool = Field(..., description="Whether CLIP pipeline is loaded")
    flux_loaded: bool = Field(..., description="Whether FLUX + LoRA pipeline is loaded")
    val_worker_running: bool = Field(..., description="Whether ArcFace identity validation worker is alive")
    gpu_free_vram_gb: Optional[float] = Field(None, description="Free GPU VRAM in GB")
    gpu_total_vram_gb: Optional[float] = Field(None, description="Total GPU VRAM in GB")

class Step1Result(BaseModel):
    key: str
    user_path: str
    crop_path: str
    face_box: List[int]
    faces: int
    background_faces_ignored: int
    gender: str
    gender_conf: float
    gender_src: str
    glasses: bool
    glasses_source: str
    hijab: bool
    beard: bool
    avatar: str

class GenerationMetadata(BaseModel):
    avatar: str
    gender: str
    glasses: bool
    hijab: bool
    beard: bool
    id_sim: Optional[float] = None
    jaw_diff: Optional[float] = None
    score: Optional[float] = None
    visor: Optional[str] = None
    visor_score: Optional[float] = None
    tries: int
    best_try: int
    stopped: str
    seconds: float
    step1_s: Optional[float] = None
    gen_s: float
    val_s: float
    lora_mult: float
