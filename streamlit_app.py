"""Streamlit client for the deployed FracAtlas FastAPI service."""

import base64
from typing import Any

import requests
import streamlit as st


BACKEND_URL = "https://detecting-abnormal-bone-fracture-using-buhz.onrender.com"
REQUEST_TIMEOUT_SECONDS = 60
ACCEPTED_EXTENSIONS = ["png", "jpg", "jpeg", "tiff", "tif"]
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024


def api_error_message(response: requests.Response) -> str:
    """Return a useful FastAPI error detail without assuming a response schema."""
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        if detail:
            return str(detail)
    return f"HTTP {response.status_code}"


def get_status() -> tuple[bool, dict[str, Any] | None, str | None]:
    """Check both deployed status endpoints and return their actual responses."""
    try:
        health_response = requests.get(f"{BACKEND_URL}/health", timeout=15)
        health_response.raise_for_status()
        health = health_response.json()
    except (requests.RequestException, ValueError) as exc:
        return False, None, f"Health check failed: {exc}"

    try:
        model_response = requests.get(f"{BACKEND_URL}/model/status", timeout=15)
        model_response.raise_for_status()
        return True, model_response.json(), None
    except (requests.RequestException, ValueError) as exc:
        return True, None, f"Model-status check failed: {exc}"


def show_evaluation_summary() -> None:
    """Display evaluation values only when the backend says they exist."""
    try:
        response = requests.get(f"{BACKEND_URL}/evaluation/summary", timeout=15)
        response.raise_for_status()
        summary = response.json()
    except (requests.RequestException, ValueError) as exc:
        st.warning(f"Could not retrieve evaluation summary: {exc}")
        return

    if not summary.get("has_evaluation_data"):
        st.info(summary.get("message", "No evaluation data is available from the backend."))
        return

    st.success(summary.get("message", "Evaluation data retrieved from the backend."))
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        display_metrics = {
            key.replace("_", " ").title(): value
            for key, value in metrics.items()
            if value is not None
        }
        if display_metrics:
            st.json(display_metrics)

    confusion_matrix = summary.get("confusion_matrix")
    if isinstance(confusion_matrix, dict):
        st.caption("Confusion matrix")
        st.dataframe(confusion_matrix.get("matrix", []), use_container_width=True)


def show_explanation(uploaded_file: Any) -> None:
    """Request and show Grad-CAM assets only when the deployed API returns them."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/explain",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not response.ok:
            st.warning(f"Explanation unavailable: {api_error_message(response)}")
            return
        explanation = response.json()
    except requests.Timeout:
        st.warning("Explanation request timed out. Please try again.")
        return
    except (requests.RequestException, ValueError) as exc:
        st.warning(f"Explanation request failed: {exc}")
        return

    overlay = explanation.get("overlay_base64")
    heatmap = explanation.get("heatmap_base64")
    if overlay or heatmap:
        st.subheader("Grad-CAM explanation")
        columns = st.columns(2)
        for column, label, image_data in zip(columns, ["Overlay", "Heatmap"], [overlay, heatmap]):
            if isinstance(image_data, str) and "," in image_data:
                column.image(base64.b64decode(image_data.split(",", 1)[1]), caption=label)
        if explanation.get("focus_regions"):
            st.caption("Focus regions returned by the backend")
            st.json(explanation["focus_regions"])


st.set_page_config(page_title="FracAtlas Screening", page_icon="🩻", layout="centered")
st.title("FracAtlas X-ray Bone Fracture Screening")
st.write("A Streamlit client for the deployed FracAtlas ResNet-18 research service.")

st.subheader("Backend Status")
st.code(BACKEND_URL, language=None)
if st.button("Refresh backend status"):
    st.rerun()

backend_connected, model_status, status_error = get_status()
if backend_connected:
    st.success("Connected / healthy")
else:
    st.error("Disconnected / unavailable")
    st.caption(status_error)

checkpoint_ready = False
if model_status is None:
    st.warning(status_error or "Model status is unavailable; screening is disabled.")
else:
    checkpoint_loaded = model_status.get("checkpoint_loaded") is True
    ready_for_inference = model_status.get("ready_for_inference") is True
    checkpoint_ready = checkpoint_loaded and ready_for_inference
    if checkpoint_ready:
        st.success("Model ready: trained checkpoint loaded")
    else:
        st.error("Model not ready: trained checkpoint is not loaded for inference")
    st.caption(f"Checkpoint path: {model_status.get('checkpoint_path', 'Not reported')}")
    if model_status.get("error_message"):
        st.warning(str(model_status["error_message"]))

with st.expander("Evaluation summary (from backend)"):
    if backend_connected:
        show_evaluation_summary()
    else:
        st.info("Evaluation summary is unavailable while the backend is disconnected.")

st.subheader("Upload an X-ray image")
uploaded_file = st.file_uploader(
    "Accepted formats: PNG, JPG, JPEG, TIFF, TIF (maximum 25 MB)",
    type=ACCEPTED_EXTENSIONS,
)

valid_file = uploaded_file is not None and uploaded_file.size <= MAX_UPLOAD_SIZE_BYTES
if uploaded_file is not None and not valid_file:
    st.error("The selected file exceeds the backend's 25 MB upload limit.")
if uploaded_file is not None and valid_file:
    st.image(uploaded_file, caption=f"Uploaded image: {uploaded_file.name}", use_container_width=True)

run_screening = st.button(
    "Run Screening",
    type="primary",
    disabled=not (backend_connected and checkpoint_ready and valid_file),
)

if run_screening and uploaded_file is not None:
    try:
        with st.spinner("Running screening on the deployed backend..."):
            response = requests.post(
                f"{BACKEND_URL}/screen",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        if not response.ok:
            st.error(f"Screening failed: {api_error_message(response)}")
        else:
            result = response.json()
            st.subheader("Screening result")
            st.success(f"Prediction: {result['predicted_class']}")
            st.metric("Confidence", f"{float(result['confidence']) * 100:.2f}%")
            if result.get("class_probabilities"):
                st.caption("Class probabilities returned by the backend")
                st.json(result["class_probabilities"])
            if result.get("quality_analysis"):
                with st.expander("Image-quality analysis"):
                    st.json(result["quality_analysis"])
            if result.get("inference_time_ms") is not None:
                st.caption(f"Inference time: {result['inference_time_ms']} ms")
            if result.get("disclaimer"):
                st.warning(result["disclaimer"])
            show_explanation(uploaded_file)
    except requests.Timeout:
        st.error("Screening request timed out. The deployed backend may be waking up; please try again.")
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        st.error(f"Screening request failed: {exc}")

st.divider()
st.caption(
    "Research prototype only — not a clinically validated diagnostic tool. "
    "Do not use this application for medical diagnosis or treatment decisions."
)
