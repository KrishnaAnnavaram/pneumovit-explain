"""Streamlit app (optional extra 'app'): `pneumovit-explain app --model models/bundle.joblib`.

The bundle and its nearest-neighbour index load once (st.cache_resource). Uploads stay in
memory and are never written to disk. The device is chosen automatically (CPU fallback).
"""

from __future__ import annotations

import io
import os

import numpy as np


def run() -> None:  # pragma: no cover - needs streamlit
    import streamlit as st
    from PIL import Image

    from pneumovit_explain import BANNER
    from pneumovit_explain.config import Settings
    from pneumovit_explain.explain import narrative
    from pneumovit_explain.pipeline import load_bundle

    st.set_page_config(page_title="pneumovit-explain", layout="wide")
    st.error(BANNER)
    settings = Settings.from_env()
    model_path = os.environ.get("PNEUMOVIT_BUNDLE", str(settings.model_dir / "bundle.joblib"))

    @st.cache_resource
    def get_bundle(path: str):
        return load_bundle(path)

    try:
        bundle = get_bundle(model_path)
    except (FileNotFoundError, TypeError) as exc:
        st.warning(f"No bundle at {model_path}: {exc}. Run `pneumovit-explain train` first.")
        return
    client = narrative.make_client(settings)
    upload = st.file_uploader("Chest X-ray (PNG or JPEG). Do not upload identifiable patient images.",
                              type=["png", "jpg", "jpeg"])
    if upload is None:
        return
    img = np.asarray(Image.open(io.BytesIO(upload.getvalue())).convert("L"), dtype=np.float32) / 255.0
    result = bundle.analyze(img, k=settings.k_neighbours, client=client, send_image=settings.llm_send_image)
    left, right = st.columns(2)
    with left:
        st.image(result["image"], caption="Input (resized)", clamp=True)
        heat = np.clip(result["heat"], 0, None)
        st.image(heat / heat.max() if heat.max() > 0 else heat, caption="Occlusion evidence for PNEUMONIA",
                 clamp=True)
    with right:
        st.metric("Calibrated pneumonia probability", f"{result['probability']:.2f}")
        st.write(f"Operating threshold {result['threshold']:.2f}: **{result['flag']}**")
        s = result["neighbours_summary"]
        st.write(f"Most similar training images: {s['pneumonia']} PNEUMONIA, {s['normal']} NORMAL")
        st.table([{k: v for k, v in nb.items()} for nb in result["neighbours"]])
        st.info(result["summary"]["text"])
        st.caption(f"Summary source: {result['summary']['source']}")


if __name__ == "__main__":  # pragma: no cover
    run()
