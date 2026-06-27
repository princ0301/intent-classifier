from pathlib import Path

import yaml

def load_config(config_name: str) -> dict:
    config_path = Path("configs") / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)