import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    result = {}
    for config in configs:
        if config:
            result.update(config)
    return result


def get_charset(charset_path: Optional[Path] = None) -> str:
    """Load character set from YAML file."""
    if charset_path is None:
        charset_path = Path(__file__).parent.parent.parent / "configs" / "charset_aurebesh.yaml"
    
    config = load_config(charset_path)
    return config['vocab']