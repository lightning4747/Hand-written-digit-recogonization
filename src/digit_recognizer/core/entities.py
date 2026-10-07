from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class PredictionResult:
    """Domain model representing a digit classification result."""
    digit: int
    confidence: float
    probabilities: List[float]


class ModelInferenceError(Exception):
    """Raised when model inference fails."""
    pass


class InvalidImageError(ValueError):
    """Raised when input image data is invalid or empty."""
    pass
