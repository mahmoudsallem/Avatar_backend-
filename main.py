import sys
from pathlib import Path

# Ensure working directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import FastAPI app instance from app.py
from app import app
