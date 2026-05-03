"""Entry point for local debugging. Loads .env then runs training."""

import os
from src.train import main

def load_env(env_path: str = ".env") -> None:
    """Load key=value pairs from a .env file into os.environ."""
    if not os.path.exists(env_path):
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key not in os.environ:
                os.environ[key] = value


if __name__ == "__main__":
    load_env()
    main()