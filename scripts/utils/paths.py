from pathlib import Path
from datetime import datetime
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def get_run_id(prefix: str = "run") -> str:
    """Generate unique run ID with timestamp."""
    return f"{prefix}_{get_timestamp()}"