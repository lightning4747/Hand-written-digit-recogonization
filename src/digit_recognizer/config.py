from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration settings."""
    model_path: Path = Path("model/handwritten.tflite")
    prediction_dir: Path = Path("prediction")
    image_width: int = 28
    image_height: int = 28
    canvas_size: int = 200
    page_title: str = "Handwritten Digit Recognition"
