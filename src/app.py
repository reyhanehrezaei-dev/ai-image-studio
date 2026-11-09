"""AI Image Studio: Gradio + Diffusers.

Standalone version extracted from the supplied project notebook.
Requires locally available model weights and sufficient RAM/VRAM.
"""

# Import libraries used for GPU cleanup, image processing, diffusion pipelines, and the Gradio UI.
import gc
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageChops
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)


# Model IDs and runtime configuration.
BASE_MODEL = "Lykon/dreamshaper-8"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"
INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# Cache the currently loaded pipeline so only one large model stays in memory.
current_pipe = None
current_name = None


# Free the active diffusion pipeline and release GPU memory.
def clear_gpu():
    global current_pipe, current_name
    if current_pipe is not None:
        try:
            del current_pipe
        except Exception:
            pass
    current_pipe = None
    current_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Apply shared scheduler and memory optimizations to a diffusion pipeline.
def _prepare_pipe(pipe, use_attention_slicing: bool = True):
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Important: IP-Adapter must be loaded before enabling attention slicing.
    # Some diffusers versions raise SlicedAttnProcessor errors if the order is reversed.
    if use_attention_slicing:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    pipe = pipe.to(DEVICE)
    return pipe


# Lazily load and cache the requested diffusion pipeline.
def get_pipe(kind: str):
    global current_pipe, current_name

    if current_pipe is not None and current_name == kind:
        return current_pipe

    clear_gpu()

    if kind == "txt2img":
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = _prepare_pipe(pipe)

    elif kind in ["style_prompt", "style_reference"]:
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=DTYPE)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Load IP-Adapter before attention slicing for reference-based style transfer.
        if kind == "style_reference":
            pipe.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder="models",
                weight_name=IP_ADAPTER_WEIGHT,
            )

        # Disable attention slicing in the style pipeline to avoid SlicedAttnProcessor issues.
        pipe = _prepare_pipe(pipe, use_attention_slicing=False)

    elif kind == "inpaint":
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            INPAINT_MODEL,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = _prepare_pipe(pipe)

    else:
        raise ValueError(f"Unknown pipe kind: {kind}")

    current_pipe = pipe
    current_name = kind
    return pipe


# Resize an image to a square canvas while preserving aspect ratio.
def resize_square(img: Image.Image, size: int = 512, fill=(255, 255, 255)) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), fill)
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


# Build a Canny edge map for ControlNet structure guidance.
def make_canny_image(img: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    img = resize_square(img, 512)
    np_img = np.array(img)
    edges = cv2.Canny(np_img, low_threshold, high_threshold)
    edges = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


# Extract the base image and painted mask from the Gradio ImageEditor value.
def extract_editor_image_and_mask(editor_value: dict[str, Any] | None):
    if editor_value is None:
        return None, None, "اول یک تصویر آپلود کن."

    image = editor_value.get("background")
    layers = editor_value.get("layers") or []

    if image is None:
        return None, None, "اول یک تصویر آپلود کن."
    if not isinstance(image, Image.Image):
        return None, None, "فرمت تصویر معتبر نیست."

    image = resize_square(image, 512)
    mask = Image.new("L", image.size, 0)

    for layer in layers:
        if layer is None or not isinstance(layer, Image.Image):
            continue
        layer = layer.convert("RGBA").resize(image.size)
        alpha = layer.getchannel("A")
        mask = ImageChops.lighter(mask, alpha)

    mask = mask.point(lambda p: 255 if p > 0 else 0)

    if mask.getbbox() is None:
        return image, None, "روی تصویر با Brush ماسک بکش."

    return image, mask, None


# Format status messages shown under each output.
def status(msg, good=True):
    icon = "✅" if good else "❌"
    return f"{icon} {msg}"


# Generate a new image from a text prompt.
def generate_txt2img(prompt, negative_prompt, steps, guidance_scale):
    if not str(prompt or "").strip():
        return None, status("پرامپت را بنویس.", False)

    pipe = get_pipe("txt2img")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        height=512,
        width=512,
    ).images[0]
    return image, status("تصویر ساخته شد.")


# Apply prompt-based or reference-based style transfer to an uploaded image.
def generate_style(content_image, style_reference_image, prompt, negative_prompt,
                   strength, steps, guidance_scale, style_scale, control_scale):
    if content_image is None:
        return None, status("اول تصویر اصلی را آپلود کن.", False)
    if not str(prompt or "").strip():
        return None, status("پرامپت سبک را بنویس.", False)

    init_image = resize_square(content_image, 512)
    control_image = make_canny_image(content_image)

    # Faster mode: prompt + ControlNet without a style reference image.
    if style_reference_image is None:
        pipe = get_pipe("style_prompt")
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=init_image,
            control_image=control_image,
            strength=float(strength),
            controlnet_conditioning_scale=float(control_scale),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
        ).images[0]
        return out, status("تغییر سبک فقط با پرامپت و ControlNet انجام شد.")

    # Higher-quality mode: prompt + ControlNet + IP-Adapter with a style reference image.
    style_image = resize_square(style_reference_image, 512)
    pipe = get_pipe("style_reference")

    try:
        pipe.set_ip_adapter_scale(float(style_scale))
    except Exception:
        pass

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=init_image,
        control_image=control_image,
        ip_adapter_image=style_image,
        strength=float(strength),
        controlnet_conditioning_scale=float(control_scale),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
    ).images[0]
    return out, status("تغییر سبک با عکس مرجع انجام شد.")


# Fill the painted mask area using the inpainting pipeline.
def generate_inpaint(editor_value, prompt, negative_prompt, strength, steps, guidance_scale):
    if not str(prompt or "").strip():
        return None, status("پرامپت را بنویس.", False)

    image, mask, err = extract_editor_image_and_mask(editor_value)
    if err:
        return None, status(err, False)

    pipe = get_pipe("inpaint")
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=image,
        mask_image=mask,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
    ).images[0]
    return out, status("Inpainting انجام شد.")


# CSS section: defines the dark visual theme, layout cards, tabs, buttons, image panels, and tooltips.
CSS = """
/* CSS theme variables and global background. */
:root {
    --bg: #0f1020;
    --surface: rgba(15, 23, 42, 0.78);
    --border: rgba(255, 255, 255, 0.14);
    --text: #f8fafc;
    --muted: rgba(248, 250, 252, 0.68);
    --accent-a: #8b5cf6;
    --accent-b: #ec4899;
    --accent-c: #fb923c;
}

html, body, #root, .app, gradio-app {
    background: #0f1020 !important;
}

.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 24px !important;
    background:
        radial-gradient(circle at top left, rgba(139, 92, 246, 0.34), transparent 30%),
        radial-gradient(circle at top right, rgba(236, 72, 153, 0.24), transparent 28%),
        linear-gradient(135deg, #0f1020 0%, #111827 52%, #18181b 100%) !important;
    color: var(--text) !important;
}

.contain, .main, .wrap, .wrap-inner, main, .gradio-container > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Hero banner layout and visual style. */
.hero {
    padding: 30px;
    border-radius: 28px;
    margin-bottom: 26px;
    background:
        linear-gradient(135deg, rgba(139, 92, 246, 0.34), rgba(236, 72, 153, 0.20)),
        rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.16);
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.26);
}

.hero h1 {
    margin: 0 !important;
    font-size: 42px !important;
    line-height: 1.05 !important;
    letter-spacing: -0.04em;
    color: white !important;
}

.hero p {
    margin-top: 12px !important;
    max-width: 820px;
    color: rgba(255, 255, 255, 0.78) !important;
    font-size: 16px !important;
}

.badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 18px;
}

.badge {
    padding: 8px 13px;
    border-radius: 999px;
    font-size: 13px;
    color: white;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.16);
}

/* Remove extra outer frames around Gradio containers. */
.main-card {
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    overflow: visible !important;
}

.main-card > div,
.main-card .tabitem,
.main-card [role="tabpanel"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    overflow: visible !important;
}

/* Style the main tabs panel as one large glass card. */
.main-card .tabs {
    padding: 34px !important;
    border-radius: 32px !important;
    background: rgba(15, 23, 42, 0.74) !important;
    border: none !important;
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22) !important;
    overflow: visible !important;
}

/* Tab navigation bar and selected tab state. */
.main-card [role="tablist"],
.main-card .tab-nav {
    background: rgba(15, 23, 42, 0.88) !important;
    border-radius: 28px 28px 0 0 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.14) !important;
    padding: 14px 18px 0 18px !important;
    margin: 0 0 30px 0 !important;
    gap: 14px !important;
    min-height: 76px !important;
    align-items: flex-end !important;
    overflow: hidden !important;
}

.main-card [role="tab"],
.main-card .tab-nav button {
    min-height: 58px !important;
    padding: 0 36px !important;
    font-size: 20px !important;
    line-height: 1.2 !important;
    border-radius: 24px 24px 0 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: rgba(255, 255, 255, 0.68) !important;
    background: transparent !important;
    font-weight: 800 !important;
}

.main-card [role="tab"][aria-selected="true"],
.main-card .tab-nav button.selected,
.main-card .tab-nav button[aria-selected="true"] {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.10) !important;
    border-bottom: 4px solid var(--accent-c) !important;
}

.main-card [role="tabpanel"] {
    padding: 0 !important;
}

/* Header card used at the top of each tool tab. */
.tool-header {
    padding: 24px 30px !important;
    margin-bottom: 26px !important;
    border-radius: 26px !important;
    background:
        linear-gradient(135deg, rgba(139, 92, 246, 0.20), rgba(236, 72, 153, 0.10)),
        rgba(30, 27, 75, 0.72) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.18) !important;
}

.tool-header h2 {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 36px !important;
    line-height: 1.2 !important;
    color: #ffffff !important;
    letter-spacing: -0.03em;
}

/* Shared card styles for input controls, outputs, and info blocks. */
.input-card, .output-card, .info-card {
    padding: 16px !important;
    border-radius: 24px !important;
    background: rgba(15, 23, 42, 0.64) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: var(--text) !important;
    overflow: visible !important;
}

.output-card {
    min-height: 560px;
}

.main-card h1, .main-card h2, .main-card h3, .main-card p,
.main-card label, .main-card span, .main-card div, .main-card .markdown {
    color: var(--text) !important;
}

.main-card p, .main-card .prose, .main-card .markdown p {
    color: var(--muted) !important;
}

.main-card .block, .input-card .block, .output-card .block {
    background: rgba(15, 23, 42, 0.50) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 18px !important;
    color: var(--text) !important;
    overflow: visible !important;
}

.main-card .form, .input-card .form, .output-card .form,
.main-card fieldset, .main-card .panel {
    background: transparent !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
    color: var(--text) !important;
    overflow: visible !important;
}

/* Form fields, labels, sliders, image frames, and buttons. */
.main-card textarea,
.main-card input,
.main-card input[type="text"],
.main-card input[type="number"] {
    background: rgba(2, 6, 23, 0.58) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    border-radius: 16px !important;
    box-shadow: none !important;
}

.main-card textarea::placeholder,
.main-card input::placeholder {
    color: rgba(255, 255, 255, 0.46) !important;
}

.main-card label,
.main-card .label-wrap,
.main-card .block-title,
.main-card .block-label {
    background: rgba(15, 23, 42, 0.90) !important;
    color: rgba(255, 255, 255, 0.84) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
    font-weight: 700 !important;
}

.main-card .accordion,
.main-card details {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 18px !important;
    color: var(--text) !important;
    overflow: visible !important;
}

.main-card summary {
    color: #ffffff !important;
    font-weight: 800 !important;
}

.main-card input[type="range"] {
    accent-color: #fb923c !important;
}

.main-card .image-container,
.main-card .image-preview,
.main-card .empty,
.main-card .upload-container,
.main-card .dropzone,
.main-card .preview,
.main-card .file-preview,
.main-card .thumbnail-item,
.main-card .image-frame,
.main-card .image-frame > div,
.main-card .image-frame .block,
.main-card .image-frame .wrap {
    background: rgba(15, 23, 42, 0.80) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
    color: rgba(255, 255, 255, 0.78) !important;
}

.main-card .image-container .label-wrap,
.main-card .image-preview .label-wrap,
.main-card .upload-container .label-wrap,
.main-card .file-preview .label-wrap {
    background: rgba(15, 23, 42, 0.95) !important;
    color: rgba(255, 255, 255, 0.84) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
}

.main-card .image-container button,
.main-card .image-preview button,
.main-card .upload-container button,
.main-card .icon-button {
    background: rgba(15, 23, 42, 0.82) !important;
    color: rgba(255, 255, 255, 0.88) !important;
    border-color: rgba(255, 255, 255, 0.14) !important;
}

.main-card canvas,
.main-card img,
.main-card video {
    background: rgba(15, 23, 42, 0.82) !important;
}

.image-frame,
.image-frame .block,
.image-frame img,
.image-frame canvas,
.image-frame video {
    border-radius: 18px !important;
    overflow: hidden !important;
}

.status-box {
    padding: 12px 14px !important;
    border-radius: 14px !important;
    background: rgba(34, 197, 94, 0.12) !important;
    border: 1px solid rgba(34, 197, 94, 0.22) !important;
    color: #dcfce7 !important;
}

button.primary {
    border-radius: 14px !important;
    min-height: 44px !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #8b5cf6, #ec4899) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 14px 32px rgba(139, 92, 246, 0.28) !important;
}

button.primary:hover {
    filter: brightness(1.08);
}

button.secondary {
    border-radius: 14px !important;
    min-height: 44px !important;
    font-weight: 700 !important;
}

.main-card svg {
    color: rgba(255, 255, 255, 0.78) !important;
    fill: currentColor !important;
}

.main-card .gap,
.main-card .compact,
.main-card .padded,
.main-card .unpadded {
    background: transparent !important;
}

footer {
    visibility: hidden;
}

/* Hide the default ImageEditor layer controls to keep the inpainting UI simple. */
/* Inpainting editor overrides: hide layer controls and add a brush instruction. */
#inpaint_editor_box {
    position: relative !important;
    overflow: hidden !important;
}

#inpaint_editor_box [class*="layer"],
#inpaint_editor_box [class*="Layer"],
#inpaint_editor_box button[title*="layer" i],
#inpaint_editor_box button[aria-label*="layer" i],
#inpaint_editor_box [aria-label*="layer" i] {
    display: none !important;
    visibility: hidden !important;
}

#inpaint_editor_box::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 28px;
    background: rgba(15, 23, 42, 0.96) !important;
    z-index: 999 !important;
    pointer-events: none;
}

/* Hover tooltip system for explaining prompts, sliders, and advanced controls. */
/* Tooltip trigger styling for each interactive control. */
#tip_txt_prompt,
#tip_txt_negative,
#tip_txt_steps,
#tip_txt_guidance,
#tip_style_prompt,
#tip_style_negative,
#tip_style_strength,
#tip_style_steps,
#tip_style_guidance,
#tip_style_scale,
#tip_control_scale,
#tip_inpaint_prompt,
#tip_inpaint_negative,
#tip_inpaint_strength,
#tip_inpaint_steps,
#tip_inpaint_guidance {
    position: relative !important;
    overflow: visible !important;
    z-index: 50 !important;
}

/* Tooltip text shown when users hover over controls. */
#tip_txt_prompt:hover::after { content: "Describe what you want the model to generate."; }
#tip_txt_negative:hover::after { content: "Describe things you want the model to avoid, such as blur, distortion, or low quality."; }
#tip_txt_steps:hover::after { content: "Controls generation quality and speed. Higher values are slower but may improve detail."; }
#tip_txt_guidance:hover::after { content: "Controls how strongly the model follows your prompt. Higher values follow the prompt more strictly."; }
#tip_style_prompt:hover::after { content: "Describe the artistic style you want to apply to the content image."; }
#tip_style_negative:hover::after { content: "Describe unwanted problems such as blur, distortion, artifacts, or changed composition."; }
#tip_style_strength:hover::after { content: "Controls how much the original image changes. Higher values create stronger style changes but may alter the subject."; }
#tip_style_steps:hover::after { content: "Controls quality and speed. More steps can improve detail but take longer."; }
#tip_style_guidance:hover::after { content: "Controls how strongly the model follows the style prompt."; }
#tip_style_scale:hover::after { content: "Controls how much the style reference image affects the result. Only matters when a reference image is provided."; }
#tip_control_scale:hover::after { content: "Controls how strongly the original image structure is preserved using ControlNet."; }
#tip_inpaint_prompt:hover::after { content: "Describe what should appear inside the painted mask area."; }
#tip_inpaint_negative:hover::after { content: "Describe unwanted issues in the edited area."; }
#tip_inpaint_strength:hover::after { content: "Controls how much the masked area changes. Higher values create stronger edits."; }
#tip_inpaint_steps:hover::after { content: "Controls inpainting quality and speed. More steps can improve detail but take longer."; }
#tip_inpaint_guidance:hover::after { content: "Controls how strongly the model follows the inpainting prompt."; }

#tip_txt_prompt:hover::after,
#tip_txt_negative:hover::after,
#tip_txt_steps:hover::after,
#tip_txt_guidance:hover::after,
#tip_style_prompt:hover::after,
#tip_style_negative:hover::after,
#tip_style_strength:hover::after,
#tip_style_steps:hover::after,
#tip_style_guidance:hover::after,
#tip_style_scale:hover::after,
#tip_control_scale:hover::after,
#tip_inpaint_prompt:hover::after,
#tip_inpaint_negative:hover::after,
#tip_inpaint_strength:hover::after,
#tip_inpaint_steps:hover::after,
#tip_inpaint_guidance:hover::after {
    position: absolute;
    left: 16px;
    bottom: calc(100% + 12px);
    z-index: 999999;
    width: max-content;
    max-width: 340px;
    padding: 12px 14px;
    background: rgba(2, 6, 23, 0.97);
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 14px;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45);
    font-size: 13px;
    line-height: 1.45;
    font-weight: 600;
    white-space: normal;
    pointer-events: none;
}

#tip_txt_prompt:hover::before,
#tip_txt_negative:hover::before,
#tip_txt_steps:hover::before,
#tip_txt_guidance:hover::before,
#tip_style_prompt:hover::before,
#tip_style_negative:hover::before,
#tip_style_strength:hover::before,
#tip_style_steps:hover::before,
#tip_style_guidance:hover::before,
#tip_style_scale:hover::before,
#tip_control_scale:hover::before,
#tip_inpaint_prompt:hover::before,
#tip_inpaint_negative:hover::before,
#tip_inpaint_strength:hover::before,
#tip_inpaint_steps:hover::before,
#tip_inpaint_guidance:hover::before {
    content: "";
    position: absolute;
    left: 32px;
    bottom: calc(100% + 4px);
    z-index: 999999;
    border-width: 8px 8px 0 8px;
    border-style: solid;
    border-color: rgba(2, 6, 23, 0.97) transparent transparent transparent;
    pointer-events: none;
}

/* Responsive adjustments for smaller screens. */
@media (max-width: 800px) {
    .gradio-container { padding: 14px !important; }
    .hero { padding: 22px; }
    .hero h1 { font-size: 30px !important; }
    .main-card [role="tab"], .main-card .tab-nav button {
        font-size: 16px !important;
        padding: 0 18px !important;
        min-height: 48px !important;
    }
    .tool-header h2 { font-size: 26px !important; }
}

/* Move the tab bar lower inside the main panel. */
.main-card .tabs {
    padding-top: 64px !important;
}

/* Add breathing room above the tab buttons. */
.main-card [role="tablist"],
.main-card .tab-nav {
    margin-top: 0 !important;
    margin-bottom: 34px !important;
}

/* Add top spacing so the tabs panel feels less stuck to the page header. */
.main-card {
    margin-top: 24px !important;
}

/* Keep only the main tabs panel visible. */
.main-card {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.main-card > div,
.main-card .tabitem,
.main-card [role="tabpanel"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}


"""


# HTML/UI section: builds the Gradio interface and injects custom markup blocks.
with gr.Blocks(title="AI Image Studio") as demo:
    # HTML hero section shown above the tabs.
    gr.HTML(
        """
        <div class="hero">
            <h1>AI Image Studio</h1>
            <p>
                Generate images, transfer styles, and edit selected parts of an image with diffusion models on Colab GPU.
            </p>
            <div class="badges">
                <span class="badge">Text to Image</span>
                <span class="badge">Style Transfer</span>
                <span class="badge">Image Inpainting</span>
            </div>
        </div>
        """
    )

    with gr.Group(elem_classes=["main-card"]):
        with gr.Tabs():
            # Text-to-image tab: prompt inputs, generation controls, and output preview.
            with gr.Tab("Text to Image"):
                gr.HTML(
                    """
                    <div class="tool-header">
                        <h2>Text to Image</h2>
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["input-card"]):
                        txt_prompt = gr.Textbox(
                            label="Prompt",
                            lines=4,
                            value="a cute small robot sitting on a wooden desk, cinematic lighting, high quality",
                            placeholder="Describe the image you want...",
                            elem_id="tip_txt_prompt",
                        )

                        txt_negative = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value="blurry, low quality, distorted",
                            placeholder="Things you do not want in the output...",
                            elem_id="tip_txt_negative",
                        )

                        txt_steps = gr.Slider(
                            5,
                            30,
                            value=15,
                            step=1,
                            label="Steps",
                            elem_id="tip_txt_steps",
                        )

                        txt_guidance = gr.Slider(
                            1,
                            12,
                            value=7.5,
                            step=0.5,
                            label="Guidance",
                            elem_id="tip_txt_guidance",
                        )

                        txt_btn = gr.Button(
                            "Generate Image",
                            variant="primary",
                            elem_classes=["primary"],
                        )

                        txt_status = gr.Markdown("", visible=False)

                    with gr.Column(scale=1, elem_classes=["output-card"]):
                        txt_output = gr.Image(
                            label="Generated Image",
                            type="pil",
                            height=520,
                            elem_classes=["image-frame"],
                        )

            # Style-transfer tab: content image, optional style reference, controls, and output preview.
            with gr.Tab("Style Transfer"):
                gr.HTML(
                    """
                    <div class="tool-header">
                        <h2>Style Transfer</h2>
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["input-card"]):
                        style_content = gr.Image(
                            label="Content Image",
                            type="pil",
                            height=280,
                            elem_classes=["image-frame"],
                        )

                        style_ref = gr.Image(
                            label="Style Reference Image - Optional",
                            type="pil",
                            height=280,
                            elem_classes=["image-frame"],
                        )

                    with gr.Column(scale=1, elem_classes=["input-card"]):
                        style_prompt = gr.Textbox(
                            label="Style Prompt",
                            lines=3,
                            value="turn this image into a watercolor painting, soft brush strokes, pastel colors, preserve the original subject and composition",
                            placeholder="Describe the style you want...",
                            elem_id="tip_style_prompt",
                        )

                        style_negative = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value="blurry, distorted, low quality, deformed, bad anatomy, extra objects, changed composition",
                            elem_id="tip_style_negative",
                        )

                        with gr.Accordion("Advanced Settings", open=True):
                            style_strength = gr.Slider(
                                0.2,
                                0.9,
                                value=0.55,
                                step=0.05,
                                label="Transformation Strength",
                                elem_id="tip_style_strength",
                            )

                            style_steps = gr.Slider(
                                5,
                                30,
                                value=10,
                                step=1,
                                label="Steps",
                                elem_id="tip_style_steps",
                            )

                            style_guidance = gr.Slider(
                                1,
                                12,
                                value=7,
                                step=0.5,
                                label="Guidance Scale",
                                elem_id="tip_style_guidance",
                            )

                            style_scale = gr.Slider(
                                0.2,
                                1.2,
                                value=0.75,
                                step=0.05,
                                label="Style Reference Strength",
                                elem_id="tip_style_scale",
                            )

                            control_scale = gr.Slider(
                                0.2,
                                1.5,
                                value=0.8,
                                step=0.05,
                                label="Structure Preservation",
                                elem_id="tip_control_scale",
                            )

                        style_btn = gr.Button(
                            "Apply Style",
                            variant="primary",
                            elem_classes=["primary"],
                        )

                        style_status = gr.Markdown("", visible=False)

                with gr.Row():
                    with gr.Column(elem_classes=["output-card"]):
                        style_output = gr.Image(
                            label="Styled Output",
                            type="pil",
                            height=520,
                            elem_classes=["image-frame"],
                        )

            # Inpainting tab: image editor, mask brush, controls, and output preview.
            with gr.Tab("Image Inpainting"):
                gr.HTML(
                    """
                    <div class="tool-header">
                        <h2>Image Inpainting</h2>
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["input-card"]):
                        inpaint_editor = gr.ImageEditor(
                            label="Upload Image and Paint Mask",
                            type="pil",
                            height=460,
                            brush=gr.Brush(
                                default_size=30,
                                colors=["#ffffff"],
                                default_color="#ffffff",
                                color_mode="fixed",
                            ),
                            eraser=gr.Eraser(default_size=30),
                            layers=gr.LayerOptions(allow_additional_layers=False),
                            transforms=[],
                            sources=["upload", "clipboard"],
                            elem_classes=["image-frame"],
                            elem_id="inpaint_editor_box",
                        )

                    with gr.Column(scale=1, elem_classes=["input-card"]):
                        inpaint_prompt = gr.Textbox(
                            label="Prompt",
                            lines=3,
                            value="remove the object and fill the background naturally, realistic photo, matching lighting",
                            placeholder="Describe what should happen in the masked area...",
                            elem_id="tip_inpaint_prompt",
                        )

                        inpaint_negative = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value="blurry, distorted, low quality, artifacts, unrealistic",
                            elem_id="tip_inpaint_negative",
                        )

                        with gr.Accordion("Advanced Settings", open=True):
                            inpaint_strength = gr.Slider(
                                0.2,
                                1.0,
                                value=0.75,
                                step=0.05,
                                label="Strength",
                                elem_id="tip_inpaint_strength",
                            )

                            inpaint_steps = gr.Slider(
                                5,
                                30,
                                value=15,
                                step=1,
                                label="Steps",
                                elem_id="tip_inpaint_steps",
                            )

                            inpaint_guidance = gr.Slider(
                                1,
                                12,
                                value=7.5,
                                step=0.5,
                                label="Guidance Scale",
                                elem_id="tip_inpaint_guidance",
                            )

                        inpaint_btn = gr.Button(
                            "Generate Inpainting",
                            variant="primary",
                            elem_classes=["primary"],
                        )

                        inpaint_status = gr.Markdown("", visible=False)

                with gr.Row():
                    with gr.Column(elem_classes=["output-card"]):
                        inpaint_output = gr.Image(
                            label="Inpainted Output",
                            type="pil",
                            height=520,
                            elem_classes=["image-frame"],
                        )

    # Event handlers connect UI buttons to the generation functions above.
    txt_btn.click(
        fn=generate_txt2img,
        inputs=[txt_prompt, txt_negative, txt_steps, txt_guidance],
        outputs=[txt_output, txt_status],
    )

    style_btn.click(
        fn=generate_style,
        inputs=[
            style_content,
            style_ref,
            style_prompt,
            style_negative,
            style_strength,
            style_steps,
            style_guidance,
            style_scale,
            control_scale,
        ],
        outputs=[style_output, style_status],
    )

    inpaint_btn.click(
        fn=generate_inpaint,
        inputs=[
            inpaint_editor,
            inpaint_prompt,
            inpaint_negative,
            inpaint_strength,
            inpaint_steps,
            inpaint_guidance,
        ],
        outputs=[inpaint_output, inpaint_status],
    )


# Start the Gradio app with queueing enabled and the custom CSS theme applied.
demo.queue().launch(
    share=False,
    debug=False,
    css=CSS,
)

