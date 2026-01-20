from pathlib import Path
import os

DATA_PATH = Path(__file__).parents[1] / "data"
ORS_API_KEY = os.getenv("ORS_API_KEY")
