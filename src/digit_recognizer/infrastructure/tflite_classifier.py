from pathlib import Path
from typing import Union
import numpy as np

from digit_recognizer.core.entities import PredictionResult, ModelInferenceError
from digit_recognizer.core.interfaces import DigitClassifier


class TFLiteDigitClassifier(DigitClassifier):
    """
    Adapter implementation of DigitClassifier using TensorFlow Lite Interpreter.
    Significantly faster initialization and lower resource consumption than full Keras.
    """

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found at: {self.model_path}")

        try:
            # Check if standalone tflite_runtime is present, otherwise fallback to tf.lite
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter

            self._interpreter = Interpreter(model_path=str(self.model_path))
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
        except Exception as e:
            raise ModelInferenceError(f"Failed to load TFLite model from {self.model_path}: {e}") from e

    def predict(self, preprocessed_image: np.ndarray) -> PredictionResult:
        try:
            # Ensure tensor matches required input dtype (typically float32)
            input_dtype = self._input_details[0]["dtype"]
            input_data = preprocessed_image.astype(input_dtype)

            self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
            self._interpreter.invoke()

            output_data = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
            probabilities = output_data.tolist()
            predicted_digit = int(np.argmax(output_data))
            confidence = float(output_data[predicted_digit])

            return PredictionResult(
                digit=predicted_digit,
                confidence=confidence,
                probabilities=probabilities,
            )
        except Exception as e:
            raise ModelInferenceError(f"Error during TFLite model inference: {e}") from e
