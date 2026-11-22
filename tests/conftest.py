import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/medipredict_test.db")
if os.path.exists("/tmp/medipredict_test.db"):
    os.remove("/tmp/medipredict_test.db")
