import sys
from pathlib import Path

file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
sys.path.append(str(project_root))

try:
    from src.taxipred.backend.api import app
except Exception as e:
    print(f"FAILED TO IMPORT APP: {e}")
    raise e

handler = app