from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from PIL import Image

from digit_recognizer.core.entities import PredictionResult


class ImagePreprocessor(ABC):
    """Port: Defines image preprocessing contract."""

    @abstractmethod
    def preprocess(self, image_data: Any) -> np.ndarray:
        """Transforms raw image input into normalized model input tensor (1, 28, 28, 1)."""
        pass


class DigitClassifier(ABC):
    """Port: Defines digit prediction contract."""

    @abstractmethod
    def predict(self, preprocessed_image: np.ndarray) -> PredictionResult:
        """Generates prediction result for normalized image input."""
        pass


class ImageRepository(ABC):
    """Port: Optional persistence for saving prediction images."""

    @abstractmethod
    def save(self, image: Image.Image, filename: str) -> str:
        """Saves an image and returns the stored path."""
        pass
