from typing import Any, Optional
from PIL import Image

from digit_recognizer.core.entities import PredictionResult, InvalidImageError
from digit_recognizer.core.interfaces import DigitClassifier, ImagePreprocessor, ImageRepository


class RecognizeDigitUseCase:
    """
    Use case orchestrating image preprocessing, optional persistence, and model classification.
    Adheres to the Single Responsibility Principle and Dependency Inversion Principle.
    """

    def __init__(
        self,
        classifier: DigitClassifier,
        preprocessor: ImagePreprocessor,
        image_repository: Optional[ImageRepository] = None,
    ):
        self._classifier = classifier
        self._preprocessor = preprocessor
        self._image_repository = image_repository

    def execute(self, raw_image: Any, save_filename: Optional[str] = None) -> PredictionResult:
        if raw_image is None:
            raise InvalidImageError("Input image cannot be None.")

        # Persist copy if repository provided and input is a PIL image
        if self._image_repository and save_filename and isinstance(raw_image, Image.Image):
            self._image_repository.save(raw_image, save_filename)

        # Preprocess to expected model tensor format
        tensor = self._preprocessor.preprocess(raw_image)

        # Classify
        return self._classifier.predict(tensor)
