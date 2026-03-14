import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.taxipred.backend.api import app
except ImportError as e:
    print(f"Import Error: {e}")
    raise e