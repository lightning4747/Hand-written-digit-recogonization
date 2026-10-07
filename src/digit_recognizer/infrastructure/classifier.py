from pathlib import Path
from typing import Union
import numpy as np
import tensorflow as tf

from digit_recognizer.core.entities import PredictionResult, ModelInferenceError
from digit_recognizer.core.interfaces import DigitClassifier


class KerasDigitClassifier(DigitClassifier):
    """
    Adapter implementation of DigitClassifier using a TensorFlow/Keras model.
    """

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        self._model = self._load_model()

    def _load_model(self):
        try:
            return tf.keras.models.load_model(str(self.model_path))
        except Exception as e:
            raise ModelInferenceError(f"Failed to load model from {self.model_path}: {e}") from e

    def predict(self, preprocessed_image: np.ndarray) -> PredictionResult:
        try:
            predictions = self._model.predict(preprocessed_image, verbose=0)
            probabilities = predictions[0].tolist()
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_digit])

            return PredictionResult(
                digit=predicted_digit,
                confidence=confidence,
                probabilities=probabilities,
            )
        except Exception as e:
            raise ModelInferenceError(f"Error during model inference: {e}") from e
