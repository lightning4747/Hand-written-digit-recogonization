from digit_recognizer.config import AppConfig
from digit_recognizer.core.use_cases import RecognizeDigitUseCase
from digit_recognizer.infrastructure.classifier import KerasDigitClassifier
from digit_recognizer.infrastructure.preprocessor import MNISTImagePreprocessor
from digit_recognizer.infrastructure.repository import FileSystemImageRepository
from digit_recognizer.infrastructure.tflite_classifier import TFLiteDigitClassifier


def create_recognize_digit_use_case(config: AppConfig | None = None) -> RecognizeDigitUseCase:
    """Composition root / factory to assemble the domain use case with concrete adapters."""
    cfg = config or AppConfig()

    # Select adapter depending on model file format
    if cfg.model_path.suffix.lower() == ".tflite":
        classifier = TFLiteDigitClassifier(model_path=cfg.model_path)
    else:
        classifier = KerasDigitClassifier(model_path=cfg.model_path)

    preprocessor = MNISTImagePreprocessor(target_size=(cfg.image_width, cfg.image_height))
    repository = FileSystemImageRepository(storage_directory=cfg.prediction_dir)
    return RecognizeDigitUseCase(
        classifier=classifier,
        preprocessor=preprocessor,
        image_repository=repository,
    )
