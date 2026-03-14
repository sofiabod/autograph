import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv():
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()



@dataclass
class DebateConfig:
    proposer_model: str = "anthropic/claude-sonnet-4"
    challenger_model: str = "openai/gpt-4o"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    max_rounds: int = 2
    proposer_temperature: float = 0.7
    challenger_temperature: float = 0.3

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
