from pathlib import Path
from typing import Union
from PIL import Image

from digit_recognizer.core.interfaces import ImageRepository


class FileSystemImageRepository(ImageRepository):
    """
    Adapter implementation of ImageRepository to persist images to the local filesystem.
    """

    def __init__(self, storage_directory: Union[str, Path]):
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)

    def save(self, image: Image.Image, filename: str) -> str:
        destination = self.storage_directory / filename
        image.save(destination)
        return str(destination)
