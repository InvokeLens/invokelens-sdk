"""SDK configuration."""

from pydantic import BaseModel
from typing import Optional


class SDKConfig(BaseModel):
    api_key: str
    endpoint_url: str = "https://api.invokelens.com"
    transport_mode: str = "http"  # "http" or "eventbridge"
    event_bus_name: Optional[str] = None
    batch_size: int = 10
    flush_interval: float = 5.0
    max_queue_size: int = 1000
