import base64
import io
from PIL import Image
import streamlit as st

from digit_recognizer.config import AppConfig
from digit_recognizer.core.entities import InvalidImageError, ModelInferenceError
from digit_recognizer.factory import create_recognize_digit_use_case
from digit_recognizer.ui.canvas_component import render_digit_canvas


@st.cache_resource
def get_use_case():
    return create_recognize_digit_use_case()


def decode_data_url(data_url: str) -> Image.Image:
    """Decodes a base64 PNG data URL into a PIL Image."""
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    return Image.open(io.BytesIO(data))


def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
            color: #0f172a;
        }

        .stApp {
            background-color: #f8fafc;
        }

        #MainMenu, header, footer {
            visibility: hidden;
            height: 0;
        }

        .block-container {
            max-width: 820px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .header-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
        }

        .header-subtitle {
            text-align: center;
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }

        iframe {
            border: none !important;
            margin: 0 auto;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_results(result):
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; box-shadow: 0 4px 18px rgba(0,0,0,0.03); text-align: center; margin-bottom: 16px;">
            <div style="font-size: 0.78rem; font-weight: 700; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.08em;">Detected Digit</div>
            <div style="font-size: 4rem; font-weight: 800; color: #1d4ed8; line-height: 1; margin: 6px 0;">{result.digit}</div>
            <div style="font-size: 0.92rem; font-weight: 600; color: #475569;">Confidence: <span style="color: #0f172a;">{result.confidence * 100:.1f}%</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-weight: 700; font-size: 0.85rem; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;'>Confidence Distribution</p>",
        unsafe_allow_html=True,
    )

    for digit in range(10):
        prob = float(result.probabilities[digit])
        col_lbl, col_bar = st.columns([1, 4])
        with col_lbl:
            if digit == result.digit:
                st.markdown(f"**Digit {digit}**")
            else:
                st.markdown(f"<span style='color: #64748b;'>Digit {digit}</span>", unsafe_allow_html=True)
        with col_bar:
            st.progress(min(max(prob, 0.0), 1.0), text=f"{prob * 100:.1f}%")


def run_app():
    config = AppConfig()
    st.set_page_config(page_title=config.page_title, layout="centered", initial_sidebar_state="collapsed")
    apply_theme()

    st.markdown('<div class="header-title">Digit Recognizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Draw a digit (0 – 9) on the canvas. It will be predicted automatically as soon as you finish drawing!</div>', unsafe_allow_html=True)

    col_canvas, col_results = st.columns([1, 1], gap="large")

    with col_canvas:
        # Custom HTML5 canvas - sends dataURL whenever a stroke finishes
        data_url = render_digit_canvas(key="user_digit_canvas")

    with col_results:
        if data_url:
            try:
                image = decode_data_url(data_url)
                use_case = get_use_case()
                result = use_case.execute(raw_image=image, save_filename="stroke_digit.png")
                render_prediction_results(result)
            except (InvalidImageError, ModelInferenceError) as e:
                st.error(f"Inference error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.markdown(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 48px 20px; box-shadow: 0 4px 18px rgba(0,0,0,0.03); text-align: center; color: #94a3b8; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <div style="font-size: 2.6rem; margin-bottom: 10px;">✍️</div>
                    <div style="font-weight: 600; font-size: 1.05rem; color: #475569;">Ready to Predict</div>
                    <div style="font-size: 0.88rem; max-width: 200px; margin-top: 4px;">Draw a stroke on the canvas to see immediate predictions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
