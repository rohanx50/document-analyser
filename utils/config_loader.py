from pathlib import Path
import os
import yaml

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str = None) -> dict:

    env_path = os.getenv("CONFIG_PATH")
    if config_path is None :
        config_path = env_path or str(_project_root() / "config" / "config.yaml")

    path= Path(config_path)


    if not path.is_absolute():
        path = _project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    
    with open(path, 'r') as file:
        return yaml.safe_load(file)

    
if __name__ == "__main__":
    try:
        config = load_config()
        print("Configuration loaded successfully:", config)
    except Exception as e:
        print(f"Error loading configuration: {e}")