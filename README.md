# AI Image Studio

**An interactive image-generation application built with Stable Diffusion, ControlNet, IP-Adapter, and Gradio.**

[View the notebook](notebooks/AI_Image_Studio.ipynb) · [Application source](src/app.py)

## Overview

AI Image Studio is a project exploring diffusion-based image synthesis and editing. It brings together three workflows in a Gradio interface: text-to-image generation, style transformation using image references and structural guidance, and mask-based image inpainting.

## Example outputs

| Text-to-image | Style transformation | Inpainting |
|:---:|:---:|:---:|
| ![Generated robot](examples/text_to_image_result.png) | ![Stylized landscape](examples/style_transfer_result1.png) | ![Inpainted scene](examples/inpainting_result.png) |

*Outputs supplied with the original project. Input-image rights should be verified before adding additional before/after examples.*

## Features

- **Text-to-image:** prompt-based generation with Stable Diffusion.
- **Style transformation:** ControlNet (Canny) for structure guidance, with optional IP-Adapter reference conditioning.
- **Image inpainting:** masked image editing with a dedicated inpainting pipeline.
- **Interactive UI:** Gradio tabs, image inputs, configurable parameters, and previews.
- **Memory-aware model loading:** lazily switches between pipelines and clears GPU memory when needed.

## Tech stack

Python · PyTorch · Hugging Face Diffusers · Transformers · Gradio · OpenCV · NumPy · Pillow

## Models used in the original implementation

- Base model: `Lykon/dreamshaper-8`
- ControlNet: `lllyasviel/sd-controlnet-canny`
- IP-Adapter: `h94/IP-Adapter` (`ip-adapter_sd15.bin`)
- Inpainting: `runwayml/stable-diffusion-inpainting`

Model downloads may require accepting the relevant model licenses, network access, and, for gated assets, Hugging Face authentication.

## Getting started

**Requirements:** Python with the dependencies below, access to model weights, and preferably an NVIDIA GPU with CUDA. CPU execution may be significantly slower or impractical for some pipelines.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the local Gradio interface:

```bash
python src/app.py
```

Alternatively, open `notebooks/AI_Image_Studio.ipynb` in a GPU-backed notebook environment and run the cells in order. The notebook contains the original environment setup and an optional Hugging Face login prompt.

**Important:** `share=False` is set by default to avoid automatically generating a public Gradio sharing link. Remote notebook environments may need different access settings.

## Project layout

```text
ai-image-studio/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── app.py
├── notebooks/
│   └── AI_Image_Studio.ipynb
└── examples/
    ├── text_to_image_result.png
    ├── style_transfer_result1.png
    └── inpainting_result.png
```

## Current limitations / reproducibility notes

- This is a **portfolio-ready organization of the submitted coursework**, not a production-deployed service. The standalone script was syntax-checked, but the model pipelines have **not yet been re-run end to end** after restructuring.
- The supplied notebook was originally built for a specific runtime; library and model compatibility should be verified on a clean environment and known-working dependency versions pinned afterward.
- The original application explicitly disables the safety checker for some model pipelines; before public deployment, review and implement appropriate content-safety safeguards and check each model's usage terms.
- Inference requires downloading pretrained weights; model checkpoints are intentionally not committed to this repository.

## Author

**Reyhaneh Rezaei** — M.Sc. student in Artificial Intelligence, Ferdowsi University of Mashhad.

This repository documents an academic project. No software license is specified yet; add one only after deciding the terms under which others may reuse the code.
