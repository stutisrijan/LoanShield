"""Optional signature-image screening page. Requires a separately trained CNN."""
from pathlib import Path
import sys
import json
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.signature_cnn import SignatureCNN

st.title("🖊️ Document & Signature Verification")
st.markdown("Optional CNN-based signature image screening module.")
st.warning(
    "A signature classifier is not a reliable identity check by itself. This page is a research "
    "prototype and must not be used to authenticate a person or make lending decisions."
)

model_path = ROOT / "models/cnn/signature_cnn.pt"
metadata_path = ROOT / "models/cnn/signature_metadata.json"

uploaded = st.file_uploader("Upload a signature image", type=["png", "jpg", "jpeg", "bmp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Uploaded signature", use_container_width=True)
    gray = ImageOps.grayscale(image)
    gray.thumbnail((128, 128))
    canvas = Image.new("L", (128, 128), color=255)
    x = (128 - gray.width) // 2
    y = (128 - gray.height) // 2
    canvas.paste(gray, (x, y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    with right:
        st.image(canvas, caption="Preprocessed 128 × 128 grayscale input", use_container_width=True)

    if not model_path.exists():
        st.info("The CNN has not been trained yet. Image upload and preprocessing are available, but no authenticity prediction will be fabricated.")
        st.markdown("**To train the module:** place labelled images into `data/signatures/train/{forged,genuine}/` and `data/signatures/val/{forged,genuine}/`, then run:")
        st.code("python -m src.models.train_signature_cnn")
    elif st.button("Run signature image screening", type="primary"):
        try:
            import torch
            model = SignatureCNN()
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            tensor = torch.from_numpy(arr[None, None, :, :])
            with torch.no_grad():
                probability_genuine = float(torch.sigmoid(model(tensor))[0].item())
            st.metric("Model-estimated genuine-class probability", f"{probability_genuine:.1%}")
            st.caption("This score reflects only the trained CNN's output. It is not a verified identity result.")
            if probability_genuine >= 0.5:
                st.success("CNN output: closer to the Genuine class")
            else:
                st.error("CNN output: closer to the Forged class")
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                st.caption(f"Training metadata: validation accuracy={metadata.get('validation_accuracy', 'n/a')}")
        except Exception as exc:
            st.error(f"Could not run the signature CNN: {exc}")
else:
    st.info("Upload a signature image to inspect preprocessing and, once trained, run the optional CNN screening model.")

st.markdown("### Required dataset layout")
st.code(
    "data/signatures/\n"
    "├── train/\n"
    "│   ├── forged/\n"
    "│   └── genuine/\n"
    "└── val/\n"
    "    ├── forged/\n"
    "    └── genuine/"
)
st.caption("Use a properly licensed dataset and split by signer where appropriate to reduce identity leakage between training and validation.")
