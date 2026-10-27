# Keeping the Design: Geometry and Material Preservation in AI Interior Renders

AI render tools turn an interior image into a photorealistic render. Architects want the render to **keep their design**: the same walls, windows, furniture and colors. A common failure is that the model changes things it should not change.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/Mervecaliskan/keep-the-design)

🚀 **Try the live demo:** [Keep the Design on Hugging Face Spaces](https://huggingface.co/spaces/Mervecaliskan/keep-the-design). Upload an interior photo and change only the floor.

📝 **Read the full step-by-step write-up on Medium:** [Does the AI Keep Your Design?](https://medium.com/@mervcaliskaan/does-the-ai-keep-your-design-measuring-geometry-and-color-in-stable-diffusion-interior-renders-43260ec239d3)

In this project I ask for one change (*"light oak wooden floor"*) and measure how well different methods keep everything else.

![Original | Method A | Method B](results/method_comparison_1.jpg)
*Left: original. Middle: Method A (edges only), where the sofa and chairs turned white and the fireplace became a window. Right: Method B (edges + original photo), where the colors and the fireplace are kept and the floor became wood.*

## Key findings

- **Edges alone are not enough.** With ControlNet Canny only, colors and materials drifted (mean ΔE 21.9). Adding the original photo (img2img) cut this by 59%.
- **img2img has a trade-off.** The more it keeps the original, the weaker the prompt. The best balance was around strength 0.6 to 0.7.
- **Canny kept the design better than MLSD and Depth.** MLSD invented objects, Depth lost details.
- **Inpainting with a floor mask removed the trade-off.** Outside the floor almost nothing changed (ΔE 0.06), and the floor changed twice as much.
- **Metric choice matters.** CLIP scores the whole image, so it missed the local floor change; ΔE inside the mask measured it better.
- **The best method is now a live demo.** Anyone can [try it](https://huggingface.co/spaces/Mervecaliskan/keep-the-design): about 2 seconds per image on a free ZeroGPU Space.

## Experiments

| # | Question | Notebook |
|---|---|---|
| 1 | Edges only vs edges + original photo, and the `strength` trade-off | [`interior_render_preservation.ipynb`](interior_render_preservation.ipynb) |
| 2 | Which ControlNet keeps the design best (Canny, MLSD, Depth), and simple speed settings | [`02_controlnet_types_and_speed.ipynb`](02_controlnet_types_and_speed.ipynb) |
| 3 | Change only the floor with a segmentation mask (inpainting) | [`03_inpainting_floor_mask.ipynb`](03_inpainting_floor_mask.ipynb) |

Setup for all experiments: Stable Diffusion 1.5 + ControlNet, float16, free Colab T4 GPU, 768×432 images, 25 steps, fixed seed (42). One prompt for all methods, so the comparison is fair.

## How I measure

| Metric | What it tells | Better |
|---|---|---|
| **SSIM** | How similar the structure is (0 to 1) | higher |
| **Edge recall** | Share of the original edges (walls, windows, furniture) that are still in the render | higher |
| **Line recall** | Share of the original *straight* lines that are still in the render (Experiment 2) | higher |
| **Mean ΔE (CIEDE2000)** | Average color change per pixel in CIELAB space. ΔE < 2 is hard to see, ΔE > 10 is a clear change | lower |
| **ΔE outside / inside the mask** | Color change outside the floor (should be 0) and on the floor (the requested change) (Experiment 3) | lower / higher |
| **CLIP score gain** | How much more the render matches *"a room with a light oak wooden floor"* than the original photo does | higher |
| **Sec/image, peak GPU memory** | Speed and memory on a free T4 GPU (Experiment 2) | lower |

## Results

### 1. Method A vs Method B (3 rooms)

| Method | What the model gets | SSIM ↑ | Edge recall ↑ | Mean ΔE ↓ |
|---|---|---|---|---|
| A: Canny only | Only the edge lines of the room | 0.354 | 0.890 | 21.9 |
| B: Canny + img2img (strength 0.6) | The edge lines **and** the original photo | **0.567** | **0.934** | **8.9** |

Method B was better on every metric and on every room. Adding the original photo cut the color change by about **59%** and kept **93%** of the original edges.

<p>
<img src="results/method_comparison_2.jpg" width="100%">
<img src="results/method_comparison_3.jpg" width="100%">
</p>

### 2. The strength trade-off

Method B has a cost: the more it keeps the original, the weaker the prompt. I tested `strength` from 0.4 to 0.8.

![Strength trade-off](results/strength_tradeoff.png)

| Strength | Mean ΔE ↓ | CLIP gain ↑ |
|---|---|---|
| 0.4 | **7.2** | 1.08 |
| 0.5 | 7.8 | 1.37 |
| 0.6 | 8.9 | 1.52 |
| 0.7 | 10.1 | **2.45** |
| 0.8 | 12.2 | 1.66 |

- As strength goes up, colors drift more: mean ΔE rises steadily from 7.2 to 12.2.
- The prompt is applied better up to 0.7, but at 0.8 the score drops, because the model starts changing things it should not (the marble kitchen island turned into wood, paintings faded).
- **The best balance for this task is around strength 0.6 to 0.7.**

![Original | 0.4 | 0.6 | 0.8](results/strength_sweep_1.jpg)
*Original | strength 0.4 | 0.6 | 0.8. At 0.8 the marble island becomes wood.*

### 3. Which ControlNet keeps the design best? Canny vs MLSD vs Depth

Same setup as Method B (img2img, strength 0.6). Only the control image changes.

![Control maps](exp2/exp2_control_maps.jpg)
*What the model sees: original | Canny (all edges) | MLSD (straight lines only) | Depth*

| Control | SSIM ↑ | Edge recall ↑ | Line recall ↑ | Mean ΔE ↓ | CLIP gain ↑ | Sec/image |
|---|---|---|---|---|---|---|
| **Canny** | **0.566** | **0.939** | **0.785** | **8.9** | 1.52 | 5.4 |
| MLSD | 0.465 | 0.854 | 0.716 | 10.8 | **2.12** | 5.5 |
| Depth | 0.492 | 0.842 | 0.698 | 10.0 | 1.90 | 5.6 |

![Original | Canny | MLSD | Depth](exp2/exp2_comparison_2.jpg)
*Original | Canny | MLSD | Depth. With MLSD and Depth, the marble island turns into wood and the fireplace changes.*

- Against my expectation, **Canny kept the design best on every metric**, even on straight-line recall, where MLSD should have had an advantage.
- **MLSD invented objects** where there were no straight lines (a floor lamp, a pouf), because curved objects like the arc lamp are missing from its map.
- **Depth kept the layout but lost details**: a fireplace disappeared and a painting became a grid of frames. A depth map shows *where* a wall is, not *what* is on it.
- MLSD and Depth applied the prompt more, because they control the image less: the same trade-off as in Experiment 1.

**Speed and memory (one room):**

| Setting | Sec/image | Mean ΔE ↓ |
|---|---|---|
| Baseline (25 steps) | 5.7 | 7.8 |
| 15 steps | **3.8** (33% faster) | 7.9 |
| Attention slicing | 8.4 (47% slower) | 7.8 |

- 15 steps gave almost the same quality in a third less time.
- Attention slicing saved no memory here, because PyTorch 2 already uses memory-efficient attention (SDPA). It only helps on smaller GPUs or larger images, so an optimisation should be measured, not assumed.

### 4. Change only the floor: inpainting with a segmentation mask

A segmentation model (SegFormer, trained on ADE20K) finds the floor. **Method C** repaints only this mask with Canny ControlNet and puts the original pixels back everywhere else.

![Original | floor mask | Method B | Method C](exp3/exp3_comparison_2.jpg)
*Original | floor mask (red) | Method B (img2img) | Method C (inpainting)*

| Method | ΔE outside ↓ | SSIM outside ↑ | Edge recall outside ↑ | ΔE inside (floor change) ↑ | CLIP gain ↑ |
|---|---|---|---|---|---|
| B: img2img | 8.9 | 0.565 | 0.941 | 8.9 | **1.52** |
| **C: inpainting** | **0.06** | **0.997** | **0.992** | **16.6** | 0.82 |

- **Inpainting removed the trade-off.** Outside the floor almost nothing changed (ΔE 0.06, far below the visible limit of about 2), and the paintings stayed for the first time. The floor itself changed about twice as much as with Method B.
- **CLIP was misleading here.** It scored Method C lower, although its floors are clearly more wooden. CLIP looks at the whole image, so Method B's wooden ceiling raised its score. For local edits, ΔE inside the mask is a better measure.
- SegFormer found the floor well and left out the rugs. Thin objects that touch the floor, like chair legs and the lamp base, are partly inside the mask.

<p>
<img src="exp3/exp3_comparison_1.jpg" width="100%">
<img src="exp3/exp3_comparison_3.jpg" width="100%">
</p>

## Live demo

The method from Experiment 3 runs as a [Gradio app on Hugging Face Spaces](https://huggingface.co/spaces/Mervecaliskan/keep-the-design). Upload a photo, choose a floor (oak, walnut, parquet, marble, concrete, terracotta) or describe your own, and the app shows the floor mask, the result and the measurements.

![Before | after](app/demo_before_after.jpg)
*Before | after: concrete floor changed to dark walnut. Outside the floor: ΔE 0.05 (nothing changed). On the floor: ΔE 19.3. Time: 2 s on GPU.*

![Live demo](app/demo_screenshot.jpg)
*The app: upload a photo, choose a floor, see the floor mask and the result.*

- **Pipeline:** SegFormer finds the floor → Stable Diffusion 1.5 + Canny ControlNet repaints only the floor → the original pixels are put back outside the mask.
- **Runs on free hardware:** on a ZeroGPU Space it uses 25 steps (about 2 s per image). Without a GPU it switches to LCM-LoRA, which needs only 6 steps, so it still runs on a CPU.
- Code: [`app/app.py`](app/app.py)

## What went wrong and what I learned

- **Problem:** with edges only, colors and materials drifted (ΔE ≈ 22): dark sofas turned white, paintings lost their colors, a fireplace became a window, and "wooden floor" was applied to the ceiling in one room.
- **Cause:** Canny edges carry no color information, so the model invents the colors.
- **Fix:** add the original photo as img2img input. Result: ΔE ≈ 9, edge recall 93%.
- **New problem:** the prompt becomes weaker (daylight was not applied, a painting changed).
- **Measured the trade-off:** the best balance was around strength 0.6 to 0.7.
- **Tried other controls:** MLSD and Depth control the image less, so they follow the prompt more but invent or lose details. For keeping a design, Canny was the safest choice.
- **Solved with a mask:** inpainting with a segmentation mask kept everything outside the floor (ΔE 0.06) and changed the floor twice as much. The best way to keep a design is to decide *where* the model may change the image.
- **Metric choice matters:** CLIP scores the whole image and missed the local floor change. A region-based metric (ΔE inside the mask) told the real story.
- **From notebook to app:** the code that ran in Colab failed on Hugging Face because `torchvision` was missing. A different environment needs its own dependency check.

## Limits

- Only 3 rooms and one seed. The method rankings and the CLIP peak at 0.7 need more images and seeds to confirm.
- In Experiments 1 and 2, mean ΔE is measured on the whole image, so it also counts the floor, which I changed on purpose. Experiment 3 measures inside and outside the floor separately.
- Experiment 2 used strength 0.6 for all control types, a value chosen for Canny in Experiment 1. MLSD and Depth may work better at a different strength.
- In Experiment 3 the result depends on the mask: thin objects on the floor can change because they are inside it. ΔE outside is not exactly 0 because of the soft mask edge and JPEG compression.
- One room already had a wooden floor, which lowers its CLIP gain.
- Peak GPU memory (6.5 GB) includes all models loaded in Experiment 2 (three ControlNets, the depth model and CLIP), so a single pipeline needs less. Speed results are from one room.

All results are reproducible: running the notebooks again with seed 42 gives the same images and numbers.

## Next steps

- A smaller mask margin, or excluding thin objects like chair legs from the mask.
- Region-based prompt metrics, for example CLIP on the floor crop only.
- More rooms and seeds, with confidence intervals.

## How to run

**Demo:** open the [Hugging Face Space](https://huggingface.co/spaces/Mervecaliskan/keep-the-design), no setup needed.

**Notebooks:**
1. Open a notebook in Google Colab.
2. *Runtime → Change runtime type → T4 GPU*.
3. Run the cells in order and upload your own interior images when asked.

**App locally:** `pip install -r app/requirements.txt` and `python app/app.py` (a GPU is recommended).

**Tech:** Python, PyTorch, Hugging Face `diffusers` and `transformers`, Stable Diffusion 1.5, ControlNet (Canny, MLSD, Depth), SegFormer, Depth Anything V2, CLIP, LCM-LoRA, Gradio, Hugging Face Spaces (ZeroGPU), OpenCV, scikit-image, pandas, matplotlib.

## Author

**Merve Caliskan** · [GitHub](https://github.com/Mervecaliskann) · [LinkedIn](https://www.linkedin.com/in/mervecaliskann) · [Medium](https://medium.com/@mervcaliskaan)
