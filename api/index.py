import sys
from pathlib import Path

# Add the root directory to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.taxipred.backend.api import app

# Vercel needs the 'app' object to be available at the module level
# This is what Vercel actually "runs"