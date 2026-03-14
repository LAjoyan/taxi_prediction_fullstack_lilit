import os
import sys
from pathlib import Path

api_dir = Path(__file__).parent.resolve()
project_root = api_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.taxipred.backend.api import app