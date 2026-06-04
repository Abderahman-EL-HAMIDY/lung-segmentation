import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import streamlit as st
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import time
import pandas as pd
from improve_quality import ENHANCEMENT_FUNCTIONS, apply_clahe, apply_gamma

# ── 1. Device & Transforms ────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE  = (256, 256)
MEAN      = [0.485, 0.456, 0.406]
STD       = [0.229, 0.224, 0.225]

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── 2. Model Folders ──────────────────────────────────────────────────────────
OLD_MODELS_DIR = "old_models"
NEW_MODELS_DIR = "new_models"

def get_model_files(folder):
    if not os.path.exists(folder):
        return []
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.pth"))])

old_model_names = get_model_files(OLD_MODELS_DIR)
new_model_names = get_model_files(NEW_MODELS_DIR)

# ── 3. Architecture Detection ─────────────────────────────────────────────────
# Both old and new models follow the same naming convention:
#   {prefix}_{arch}_{encoder}_{dataset}_best.pth
# Examples:
#   new_models: unetpp_efficientnetb4_CXR_best.pth
#   old_models: baseline_unetpp_efficientnetb4_CXR_best.pth
#               baseline_unetpp_resnet50_COVIDQU_best.pth
#               baseline_unetpp_resnet34_CXR_best.pth
#               baseline_unet_efficientnetb4_CXR_best.pth
#               baseline_deeplabv3plus_efficientnetb4_CXR_best.pth
#               baseline_manet_efficientnetb4_CXR_best.pth

def detect_arch(filename):
    """
    Detect architecture, encoder and training dataset from filename.
    Works for both old_models (baseline_* prefix) and new_models.
    Returns: (arch_display, arch_key, encoder, train_ds)
    """
    name = filename.lower()

    # ── Encoder ───────────────────────────────────────────────────────────────
    if "efficientnetb7" in name or "efficientnet_b7" in name:
        encoder = "efficientnet-b7"
    elif "efficientnetb4" in name or "efficientnet_b4" in name:
        encoder = "efficientnet-b4"
    elif "resnet50" in name:
        encoder = "resnet50"
    elif "resnet34" in name:
        encoder = "resnet34"
    else:
        encoder = "efficientnet-b4"   # safe default

    # ── Architecture ──────────────────────────────────────────────────────────
    # Check deeplabv3plus before unet to avoid partial matches
    if "deeplabv3plus" in name or "deeplab" in name:
        arch_display = "DeepLabV3+"
        arch_key     = "deeplabv3plus"
    elif "manet" in name:
        arch_display = "MAnet"
        arch_key     = "manet"
    elif "unetpp" in name:
        arch_display = "UNet++"
        arch_key     = "unetpp"
    elif "unet" in name:
        arch_display = "UNet"
        arch_key     = "unet"
    else:
        arch_display = "UNet++"
        arch_key     = "unetpp"

    # ── Training dataset ──────────────────────────────────────────────────────
    if "covidqu" in name:
        train_ds = "COVIDQU"
    elif "_cxr" in name:
        train_ds = "CXR"
    else:
        train_ds = "Unknown"

    return arch_display, arch_key, encoder, train_ds


# Keep old names as thin wrappers so nothing else in the file breaks
def detect_new_model_arch(filename):
    return detect_arch(filename)

def detect_old_model_arch(filename):
    arch_display, arch_key, encoder, train_ds = detect_arch(filename)
    return arch_display, encoder, train_ds   # old callers expect 3-tuple


# ── 4. Model Builder ──────────────────────────────────────────────────────────
def build_smp_model(arch_key, encoder):
    """Instantiate an smp model with the correct attention settings."""
    if arch_key == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder, encoder_weights=None,
            in_channels=3, classes=1,
        )
    elif arch_key == "manet":
        return smp.MAnet(
            encoder_name=encoder, encoder_weights=None,
            in_channels=3, classes=1,
        )
    elif arch_key == "unetpp":
        return smp.UnetPlusPlus(
            encoder_name=encoder, encoder_weights=None,
            in_channels=3, classes=1,
            decoder_attention_type="scse",
        )
    else:   # unet
        return smp.Unet(
            encoder_name=encoder, encoder_weights=None,
            in_channels=3, classes=1,
            decoder_attention_type="scse",
        )


def _load_weights(model, ckpt_path):
    """Load checkpoint handling all common dict key variants."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    for key in ("model_state", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    model.load_state_dict(ckpt)
    return model


# ── 5. Model Loading ──────────────────────────────────────────────────────────
@st.cache_resource
def load_new_model(model_name):
    arch_display, arch_key, encoder, train_ds = detect_arch(model_name)
    model = build_smp_model(arch_key, encoder).to(DEVICE)
    model = _load_weights(model, os.path.join(NEW_MODELS_DIR, model_name))
    model.eval()
    return model, arch_display, encoder, train_ds


@st.cache_resource
def load_old_model(model_name):
    """
    Old models (baseline_*) follow the same naming convention as new models,
    so we use the same detect_arch() function — no hardcoded encoders.
    """
    arch_display, arch_key, encoder, train_ds = detect_arch(model_name)
    model = build_smp_model(arch_key, encoder).to(DEVICE)
    model = _load_weights(model, os.path.join(OLD_MODELS_DIR, model_name))
    model.eval()
    return model, arch_display, encoder, train_ds


# ── 6. Inference ──────────────────────────────────────────────────────────────
def predict_lung_mask(model, input_image):
    img_pil     = Image.fromarray(input_image).convert("RGB")
    img_resized = img_pil.resize(IMG_SIZE)
    img_tensor  = to_tensor(img_resized).unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        output    = model(img_tensor)
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    inference_time = time.time() - start_time

    binary_mask = (prob_mask > 0.5).astype(np.uint8) * 255

    img_cv   = np.array(img_resized)
    red_mask = np.zeros_like(img_cv)
    red_mask[:, :, 0] = binary_mask
    overlay  = cv2.addWeighted(img_cv, 0.7, red_mask, 0.3, 0)

    lung_pixels  = np.sum(binary_mask == 255)
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    coverage     = (lung_pixels / total_pixels) * 100
    confidence   = float(np.mean(prob_mask[prob_mask > 0.5])) if np.any(prob_mask > 0.5) else 0.0

    return binary_mask, overlay, prob_mask, {
        "inference_time": inference_time,
        "lung_coverage":  coverage,
        "confidence":     confidence,
    }


# ── 7. Streamlit UI ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Lung Segmentation v2", page_icon="🫁", layout="wide")
st.title("🫁 Lung X-Ray Segmentation — Augmented Dataset Models")
st.caption("New models trained on augmented CXR and COVID-QU-Ex datasets")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.info(f"Device: **{DEVICE}**")
    st.markdown(f"**New models:** {len(new_model_names)}")
    st.markdown(f"**Old models:** {len(old_model_names)}")
    st.markdown("---")

    model_source = st.radio("Model Source", ["🆕 New Models", "📦 Old Models", "⚖️ Compare Old vs New"])
    mode         = st.radio("Mode", ["🔬 Single Model", "📊 Compare All Models"])

    if mode == "🔬 Single Model":
        if model_source == "🆕 New Models":
            available = new_model_names
        elif model_source == "📦 Old Models":
            available = old_model_names
        else:
            available = new_model_names

        if available:
            selected_model = st.selectbox("Choose Model", available)
        else:
            st.error("No models found in selected folder.")

    st.markdown("---")

    # Enhancement settings
    st.header("🖼️ Image Enhancement")
    enhance_enabled = st.toggle("Enable Enhancement", value=False)

    selected_enhancements = []
    clahe_clip            = 2.0
    clahe_grid            = 8
    gamma_value           = 1.0
    denoise_strength      = 10

    if enhance_enabled:
        selected_enhancements = st.multiselect(
            "Select enhancements (applied in order)",
            list(ENHANCEMENT_FUNCTIONS.keys()),
            default=["CLAHE"]
        )
        if "CLAHE" in selected_enhancements:
            clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 10.0, 2.0, 0.5)
            clahe_grid = st.slider("CLAHE Grid Size", 2, 16, 8, 1)
        if "Gamma Correction" in selected_enhancements:
            gamma_value = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
        if "Denoise" in selected_enhancements:
            denoise_strength = st.slider("Denoise Strength", 1, 30, 10, 1)

    st.markdown("---")

    # Model listing
    if new_model_names:
        st.markdown("### 🆕 New Models")
        for name in new_model_names:
            arch, _, encoder, train_ds = detect_arch(name)
            st.markdown(f"- `{name}`")
            st.caption(f"  {arch} / {encoder} / {train_ds}")

    if old_model_names:
        st.markdown("### 📦 Old Models")
        for name in old_model_names:
            arch, _, encoder, train_ds = detect_arch(name)
            st.markdown(f"- `{name}`")
            st.caption(f"  {arch} / {encoder} / {train_ds}")


# ── Image Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload an X-Ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.info("👆 Upload an X-Ray image to get started.")
    st.stop()

input_image = np.array(Image.open(uploaded_file).convert("RGB"))

# ── Apply Enhancements ─────────────────────────────────────────────────────────
if enhance_enabled and selected_enhancements:
    enhanced_image = input_image.copy()
    for name in selected_enhancements:
        if name == "CLAHE":
            enhanced_image = apply_clahe(enhanced_image, clahe_clip, clahe_grid)
        elif name == "Gamma Correction":
            enhanced_image = apply_gamma(enhanced_image, gamma_value)
        elif name == "Denoise":
            from improve_quality import apply_denoise
            enhanced_image = apply_denoise(enhanced_image, denoise_strength)
        else:
            enhanced_image = ENHANCEMENT_FUNCTIONS[name](enhanced_image)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(np.array(Image.fromarray(input_image).resize(IMG_SIZE)), use_container_width=True)
    with col2:
        st.markdown(f"**Enhanced** ({' → '.join(selected_enhancements)})")
        st.image(np.array(Image.fromarray(enhanced_image).resize(IMG_SIZE)), use_container_width=True)
    process_image = enhanced_image
else:
    process_image = input_image

resized_input = np.array(Image.fromarray(process_image).resize(IMG_SIZE))


# ── Helper: run and display one model ─────────────────────────────────────────
def display_result(model_name, binary_mask, overlay, prob_mask, stats, arch, encoder, train_ds):
    st.markdown(f"#### `{model_name.replace('_best.pth', '').replace('.pth', '')}`")
    st.caption(f"Arch: **{arch}** | Encoder: **{encoder}** | Trained on: **{train_ds}**")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Input**");         st.image(resized_input,  use_container_width=True)
    with c2:
        st.markdown("**Mask**");          st.image(binary_mask,    use_container_width=True)
    with c3:
        st.markdown("**Overlay**");       st.image(overlay,        use_container_width=True)
    with c4:
        st.markdown("**Heatmap**")
        heatmap       = (prob_mask * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        st.image(heatmap_color, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Inference Time", f"{stats['inference_time']*1000:.1f} ms")
    m2.metric("Lung Coverage",  f"{stats['lung_coverage']:.1f}%")
    m3.metric("Avg Confidence", f"{stats['confidence']:.2%}")
    st.markdown("---")


# ── SINGLE MODEL MODE ─────────────────────────────────────────────────────────
if mode == "🔬 Single Model":
    if not available:
        st.error("No models found.")
        st.stop()

    if st.button("🚀 Segment Lungs", type="primary", use_container_width=True):
        with st.spinner(f"Running `{selected_model}`..."):
            try:
                if model_source == "🆕 New Models":
                    model, arch, encoder, train_ds = load_new_model(selected_model)
                else:
                    model, arch, encoder, train_ds = load_old_model(selected_model)
                binary_mask, overlay, prob_mask, stats = predict_lung_mask(model, process_image)
                display_result(selected_model, binary_mask, overlay, prob_mask,
                               stats, arch, encoder, train_ds)
            except Exception as e:
                st.error(f"Error: {e}")


# ── COMPARE ALL MODELS MODE ───────────────────────────────────────────────────
elif mode == "📊 Compare All Models":

    if model_source == "⚖️ Compare Old vs New":
        st.info("Will run all old models then all new models and compare side by side.")
        models_to_run = [("old", n) for n in old_model_names] + \
                        [("new", n) for n in new_model_names]
    elif model_source == "🆕 New Models":
        models_to_run = [("new", n) for n in new_model_names]
    else:
        models_to_run = [("old", n) for n in old_model_names]

    if not models_to_run:
        st.error("No models found in selected folder.")
        st.stop()

    if st.button("🚀 Run All Models", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text  = st.empty()
        all_results  = []

        for idx, (source, name) in enumerate(models_to_run):
            status_text.text(f"Running {name} ({idx+1}/{len(models_to_run)})...")
            progress_bar.progress((idx + 1) / len(models_to_run))
            try:
                if source == "new":
                    model, arch, encoder, train_ds = load_new_model(name)
                else:
                    model, arch, encoder, train_ds = load_old_model(name)
                binary_mask, overlay, prob_mask, stats = predict_lung_mask(model, process_image)
                all_results.append((name, binary_mask, overlay, prob_mask,
                                    stats, arch, encoder, train_ds, source))
            except Exception as e:
                st.error(f"Failed `{name}`: {e}")

        status_text.text("✅ Done!")
        progress_bar.empty()

        st.markdown("---")
        st.markdown("### Results")

        for name, binary_mask, overlay, prob_mask, stats, arch, encoder, train_ds, source in all_results:
            label = f"[{'NEW' if source == 'new' else 'OLD'}] {name}"
            display_result(label, binary_mask, overlay, prob_mask, stats, arch, encoder, train_ds)

        # Summary table
        st.markdown("### 📋 Summary Comparison")
        summary = []
        for name, _, _, _, stats, arch, encoder, train_ds, source in all_results:
            summary.append({
                "Source":          "🆕 New" if source == "new" else "📦 Old",
                "Model":           name.replace("_best.pth", "").replace(".pth", ""),
                "Architecture":    arch,
                "Encoder":         encoder,
                "Train DS":        train_ds,
                "Inference (ms)":  round(stats["inference_time"] * 1000, 1),
                "Lung Coverage %": round(stats["lung_coverage"], 1),
                "Avg Confidence":  round(stats["confidence"] * 100, 1),
            })

        df = pd.DataFrame(summary)
        st.dataframe(df, use_container_width=True, hide_index=True)

        best_conf = df.loc[df["Avg Confidence"].idxmax(),  "Model"]
        fastest   = df.loc[df["Inference (ms)"].idxmin(),  "Model"]
        best_cov  = df.loc[df["Lung Coverage %"].idxmax(), "Model"]

        st.success(f"**Highest confidence:** `{best_conf}`")
        st.info(f"**Fastest inference:** `{fastest}`")
        st.info(f"**Most lung coverage:** `{best_cov}`")
        st.caption(
            "Note: Without ground truth masks these metrics reflect model confidence only, "
            "not actual segmentation accuracy."
        )