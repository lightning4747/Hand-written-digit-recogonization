import os
from pathlib import Path
from typing import Optional
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent / "custom_canvas"
_digit_canvas_func = components.declare_component("digit_canvas", path=str(_COMPONENT_DIR))


def render_digit_canvas(key: str = "digit_canvas") -> Optional[str]:
    """
    Renders an HTML5 drawing canvas with black background and smooth white stroke.
    When a stroke is completed (mouseup/touchend), it automatically converts the drawing
    into a base64 PNG data URL and passes it back to Streamlit.
    """
    return _digit_canvas_func(key=key, default=None)
