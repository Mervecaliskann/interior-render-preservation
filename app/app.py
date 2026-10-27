"""Keep the Design: change only the floor of an interior photo.

Pipeline: SegFormer (ADE20K) finds the floor -> Stable Diffusion 1.5 + Canny ControlNet
repaints only the floor (inpainting) -> original pixels are put back outside the mask.
Runs on a free ZeroGPU Space (25 steps). On CPU, LCM-LoRA makes it fast enough (6 steps).
"""
import os
import time

ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))   # set by Hugging Face on ZeroGPU Spaces
if ZERO_GPU:
    import spaces                                     # must be imported before torch

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetInpaintPipeline
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

DEVICE = "cuda" if (ZERO_GPU or torch.cuda.is_available()) else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
USE_LCM = DEVICE == "cpu"                   # LCM-LoRA only on CPU, to make it fast enough
WIDTH = 768 if DEVICE == "cuda" else 512     # same size as the experiments on GPU
STEPS = 6 if USE_LCM else 25
GUIDANCE = 1.0 if USE_LCM else 7.5
SEED = 42

# ---------- models (loaded once when the Space starts) ----------
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=DTYPE)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet, torch_dtype=DTYPE,
    safety_checker=None, requires_safety_checker=False,
).to(DEVICE)
if USE_LCM:
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()

SEG_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
seg_proc = SegformerImageProcessor.from_pretrained(SEG_NAME)
seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_NAME).to(DEVICE).eval()
FLOOR_ID = [i for i, l in seg_model.config.id2label.items() if l.split(",")[0].strip() == "floor"][0]

MATERIALS = {
    "Light oak wood": "light oak wooden floor",
    "Dark walnut wood": "dark walnut wooden floor",
    "Herringbone parquet": "herringbone oak parquet floor",
    "White marble": "polished white marble floor",
    "Grey concrete": "smooth grey polished concrete floor",
    "Terracotta tiles": "terracotta tile floor",
}


# ---------- helpers ----------
def resize_keep_ratio(img: Image.Image) -> Image.Image:
    w, h = img.size
    new_w = WIDTH
    new_h = int(round(h * new_w / w / 8)) * 8          # Stable Diffusion needs multiples of 8
    return img.convert("RGB").resize((new_w, max(new_h, 64)))


def canny_map(room: Image.Image) -> Image.Image:
    edges = cv2.Canny(np.array(room), 100, 200)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def floor_mask(room: Image.Image, grow: int = 9) -> Image.Image:
    inputs = seg_proc(images=room, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    w, h = room.size
    logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    labels = logits.argmax(1)[0].cpu().numpy()
    mask = (labels == FLOOR_ID).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((grow, grow), np.uint8))
    return Image.fromarray(mask)


def overlay(room: Image.Image, mask: Image.Image) -> Image.Image:
    arr = np.array(room).copy()
    m = np.array(mask) > 0
    arr[m] = (0.5 * arr[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    return Image.fromarray(arr)


# ---------- main function ----------
def gpu(fn):
    return spaces.GPU(duration=60)(fn) if ZERO_GPU else fn


@gpu
def change_floor(image, material, custom_prompt):
    if image is None:
        raise gr.Error("Please upload an interior photo first.")
    t0 = time.perf_counter()

    room = resize_keep_ratio(image)
    mask = floor_mask(room)
    floor_share = (np.array(mask) > 0).mean()
    if floor_share < 0.01:
        raise gr.Error("I could not find a floor in this image. Try a photo where the floor is visible.")

    floor_text = custom_prompt.strip() or MATERIALS[material]
    prompt = f"photorealistic interior photo, {floor_text}, natural light"

    raw = pipe(
        prompt, negative_prompt=None if USE_LCM else "blurry, distorted, low quality, deformed",
        image=room, mask_image=mask, control_image=canny_map(room),
        width=room.width, height=room.height,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
        generator=torch.manual_seed(SEED),
    ).images[0]
    soft = pipe.mask_processor.blur(mask, blur_factor=4)
    result = pipe.image_processor.apply_overlay(soft, room, raw)   # original pixels outside the floor

    orig, rend = np.array(room), np.array(result)
    inside = np.array(mask) > 0
    de = deltaE_ciede2000(rgb2lab(orig), rgb2lab(rend))
    seconds = time.perf_counter() - t0

    report = (
        f"| Measure | Value |\n|---|---|\n"
        f"| Floor found | {floor_share:.0%} of the image |\n"
        f"| Color change **outside** the floor (ΔE, lower = design kept) | **{de[~inside].mean():.2f}** |\n"
        f"| Color change **on** the floor (ΔE, higher = floor changed) | **{de[inside].mean():.1f}** |\n"
        f"| Time | {seconds:.0f} s on {'GPU' if DEVICE == 'cuda' else 'CPU'} |\n\n"
        f"ΔE below 2 is hard to see; above 10 is a clear change."
    )
    return overlay(room, mask), result, report


# ---------- UI ----------
with gr.Blocks(title="Keep the Design") as demo:
    gr.Markdown(
        "# 🏠 Keep the Design: change only the floor\n"
        "Upload an interior photo and choose a new floor. A segmentation model finds the floor, "
        "and Stable Diffusion repaints **only the floor**, so the walls, furniture and paintings stay exactly the same.\n\n"
        "[GitHub](https://github.com/Mervecaliskann/interior-render-preservation) · "
        "[Medium article](https://medium.com/@mervcaliskaan/does-the-ai-keep-your-design-measuring-geometry-and-color-in-stable-diffusion-interior-renders-43260ec239d3)"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Interior photo")
            material = gr.Dropdown(list(MATERIALS), value="Light oak wood", label="New floor")
            custom = gr.Textbox(label="Or describe your own floor (optional)",
                                placeholder="e.g. black and white checkerboard tiles")
            btn = gr.Button("Change the floor", variant="primary")
            gr.Markdown("⏳ One image takes about 10 seconds on GPU (about a minute on CPU).")
        with gr.Column():
            mask_out = gr.Image(label="Floor found (red)")
            result_out = gr.Image(label="Result")
            report_out = gr.Markdown()
    btn.click(change_floor, inputs=[inp, material, custom], outputs=[mask_out, result_out, report_out])

demo.queue(max_size=5).launch()
