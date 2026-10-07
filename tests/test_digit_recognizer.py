import numpy as np
import pytest
from PIL import Image

from digit_recognizer.core.entities import PredictionResult, InvalidImageError
from digit_recognizer.core.interfaces import DigitClassifier, ImagePreprocessor, ImageRepository
from digit_recognizer.core.use_cases import RecognizeDigitUseCase
from digit_recognizer.infrastructure.preprocessor import MNISTImagePreprocessor


class DummyClassifier(DigitClassifier):
    def predict(self, preprocessed_image: np.ndarray) -> PredictionResult:
        return PredictionResult(
            digit=7,
            confidence=0.99,
            probabilities=[0.0] * 7 + [0.99] + [0.0] * 2,
        )


class DummyRepository(ImageRepository):
    def __init__(self):
        self.saved_files = []

    def save(self, image: Image.Image, filename: str) -> str:
        self.saved_files.append(filename)
        return filename


def test_recognize_digit_use_case():
    classifier = DummyClassifier()
    preprocessor = MNISTImagePreprocessor()
    repo = DummyRepository()
    use_case = RecognizeDigitUseCase(classifier, preprocessor, repo)

    dummy_img = Image.new("RGBA", (200, 200), color="black")
    result = use_case.execute(dummy_img, save_filename="test.png")

    assert result.digit == 7
    assert result.confidence == 0.99
    assert repo.saved_files == ["test.png"]


def test_recognize_digit_use_case_none_image():
    classifier = DummyClassifier()
    preprocessor = MNISTImagePreprocessor()
    use_case = RecognizeDigitUseCase(classifier, preprocessor)

    with pytest.raises(InvalidImageError):
        use_case.execute(None)


def test_preprocessor_output_shape():
    preprocessor = MNISTImagePreprocessor()
    dummy_array = np.zeros((200, 200, 4), dtype=np.uint8)
    tensor = preprocessor.preprocess(dummy_array)

    assert tensor.shape == (1, 28, 28, 1)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
