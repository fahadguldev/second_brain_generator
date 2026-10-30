"""FastAPI Backend Entrypoint."""
from pathlib import Path
import sys

_backend_root = Path(__file__).resolve().parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from src.main import app

__all__ = ["app"]


def main():
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
