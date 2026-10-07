"""pneumovit-explain: pneumonia classification on chest X-rays with validated selection and prediction-independent explanations.

The core package needs numpy, pandas, scipy, scikit-learn and Pillow. PyTorch, timm,
Streamlit and FAISS are optional extras that are imported lazily.
"""

__version__ = "0.1.0"

CLASSES = ("NORMAL", "PNEUMONIA")
POSITIVE = "PNEUMONIA"

BANNER = (
    "NOT A DIAGNOSTIC DEVICE. Research software trained on one public pediatric dataset. "
    "A qualified clinician must make every clinical decision."
)
