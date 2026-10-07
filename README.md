# Handwritten Digit Recognition

An interactive handwritten digit classification application built with Clean Architecture, TensorFlow Lite, and Streamlit.

## Features

- Clean Architecture Design: Separation of concerns across Core (domain entities, use cases, ports), Infrastructure (adapters for TFLite and image processing), and UI (Streamlit presentation).
- High-Performance Inference: Optimized with TensorFlow Lite (`.tflite`) for fast CPU inference and low memory consumption.
- Custom HTML5 Drawing Canvas: Native HTML5 canvas with a pitch-black background and smooth white drawing strokes.
- Automatic Stroke Inference: Converts strokes to images on completion and evaluates predictions immediately.
- Centered MNIST Preprocessing: Automatically scales and centers user drawings into a standardized 28x28 bounding box matching the MNIST dataset format.
- Comprehensive Test Suite: Includes unit tests for domain use cases and preprocessing adapters.

## Architecture

The project follows Clean Architecture principles:

```
Handwritten-Digit-Recognition/
├── App.py                               # Application entrypoint
├── model/
│   ├── handwritten.tflite               # Optimized TensorFlow Lite model
│   └── handwritten.h5                   # Original Keras model
├── src/
│   └── digit_recognizer/
│       ├── config.py                    # Application configuration
│       ├── factory.py                   # Dependency injection composition root
│       ├── core/                        # Domain entities, ports, and use cases
│       │   ├── entities.py
│       │   ├── interfaces.py
│       │   └── use_cases.py
│       ├── infrastructure/              # Concrete adapters
│       │   ├── classifier.py            # Keras classifier adapter
│       │   ├── tflite_classifier.py     # TensorFlow Lite adapter
│       │   ├── preprocessor.py          # MNIST bounding-box centering pipeline
│       │   └── repository.py            # Image persistence adapter
│       └── ui/                          # Presentation layer
│           ├── canvas_component.py      # Streamlit custom component wrapper
│           ├── custom_canvas/           # Native HTML5 canvas component
│           └── streamlit_app.py         # Streamlit UI logic and styling
└── tests/
    └── test_digit_recognizer.py         # Unit tests
```

## Setup and Installation

### Prerequisites

- Python 3.10 or 3.11

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lightning4747/Hand-written-digit-recogonization.git
   cd Hand-written-digit-recogonization
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install streamlit tensorflow pillow numpy pandas pytest
   ```

## Running the Application

Start the Streamlit application:

```bash
streamlit run App.py
```

Open your browser and navigate to:
```
http://localhost:8501
```

## Running Tests

Run the test suite using pytest:

```bash
PYTHONPATH=src pytest tests
```
