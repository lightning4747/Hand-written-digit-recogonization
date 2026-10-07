from typing import Any
import numpy as np
from PIL import Image, ImageOps

from digit_recognizer.core.entities import InvalidImageError
from digit_recognizer.core.interfaces import ImagePreprocessor


class MNISTImagePreprocessor(ImagePreprocessor):
    """
    Adapter for converting user-drawn canvas images into MNIST-compatible format:
    1. Grayscale & Inversion: Ensures bright digit on dark background.
    2. Bounding Box Cropping: Isolates the drawn digit so location on canvas doesn't skew results.
    3. Aspect-Preserving Scaling: Resizes the digit to fit within a 20x20 box (standard MNIST format).
    4. Centering by Center of Mass / Geometry: Centers the digit within a 28x28 canvas with a 4px border.
    5. Normalization: Scales pixel intensities to [0.0, 1.0].
    """

    def __init__(self, target_size: tuple[int, int] = (28, 28), inner_box_size: int = 20):
        self.target_size = target_size
        self.inner_box_size = inner_box_size

    def preprocess(self, image_data: Any) -> np.ndarray:
        if image_data is None:
            raise InvalidImageError("Cannot preprocess None image data.")

        if isinstance(image_data, np.ndarray):
            if image_data.size == 0:
                raise InvalidImageError("Input array is empty.")
            if image_data.ndim == 3 and image_data.shape[2] == 4:
                pil_img = Image.fromarray(image_data.astype(np.uint8), "RGBA")
            else:
                pil_img = Image.fromarray(image_data.astype(np.uint8))
        elif isinstance(image_data, Image.Image):
            pil_img = image_data
        else:
            raise InvalidImageError(f"Unsupported image type: {type(image_data)}")

        # Convert to grayscale
        gray_img = ImageOps.grayscale(pil_img)
        gray_arr = np.array(gray_img, dtype=np.float32)

        # In MNIST, foreground digit is white (255) and background is black (0).
        # If user drew on white background with dark ink (average brightness > 127), invert it.
        if np.mean(gray_arr) > 127:
            gray_img = ImageOps.invert(gray_img)
            gray_arr = np.array(gray_img, dtype=np.float32)

        # Find digit bounding box (pixels above threshold)
        active_coords = np.argwhere(gray_arr > 25)
        if active_coords.size == 0:
            # Blank canvas, return zeros
            return np.zeros((1, self.target_size[0], self.target_size[1], 1), dtype=np.float32)

        y_min, x_min = active_coords.min(axis=0)
        y_max, x_max = active_coords.max(axis=0) + 1
        cropped_digit = gray_img.crop((x_min, y_min, x_max, y_max))

        # Scale preserving aspect ratio so the larger dimension fits in inner_box_size (20px)
        w, h = cropped_digit.size
        scale = float(self.inner_box_size) / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized_digit = cropped_digit.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Create blank 28x28 black image and paste resized digit in the center
        centered_img = Image.new("L", self.target_size, color=0)
        pad_x = (self.target_size[1] - new_w) // 2
        pad_y = (self.target_size[0] - new_h) // 2
        centered_img.paste(resized_digit, (pad_x, pad_y))

        # Normalize to [0.0, 1.0] float32
        norm_arr = np.array(centered_img, dtype=np.float32) / 255.0
        tensor = norm_arr.reshape((1, self.target_size[0], self.target_size[1], 1))
        return tensor
