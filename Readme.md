# Advanced Art Style Transfer

## Overview
Advanced Art Style Transfer is a tool for converting your images into artistic styles using state-of-the-art image-to-image diffusion models. This repository leverages the [Diffusers](https://github.com/huggingface/diffusers) library and provides support for multiple styles, such as "best" (high-quality Stable Diffusion 2-1), "ghibli", "anime", "van_gogh", and "watercolor". The code is optimized for GPU usage and can be run on local GPUs (like a GTX 1650) or on Google Colab (with an A100 GPU).

## Features
- **Image-to-Image Style Transfer:** Transform your input images into various artistic styles.
- **Multiple Models:** Choose from different pre-trained models for distinct artistic outputs.
- **GPU Optimized:** Fully leverages GPU resources for high-quality output.
- **Configurable Parameters:** Customize transformation strength, guidance scale, and inference steps to fine-tune your results.

## Directory Structure
'''
'''
## Installation
'''
'''

### Requirements
- Python 3.8 or higher
- [PyTorch](https://pytorch.org/)
- [Diffusers](https://github.com/huggingface/diffusers)
- [Transformers](https://github.com/huggingface/transformers)
- [Accelerate](https://github.com/huggingface/accelerate)
- [xformers](https://github.com/facebookresearch/xformers)
- Matplotlib
- Pillow# Studio Ghibli-Style Image Generator using SDXL Refiner

This project leverages the `stabilityai/stable-diffusion-xl-refiner-1.0` model to transform input images into detailed, high-quality Studio Ghibli-style artwork. The model refines images with intricate details, soft pastel colors, and a magical anime aesthetic.

## Features
- **Uses SDXL Refiner:** Ensures high-quality, refined outputs
- **Optimized for Ghibli Aesthetic:** Generates anime-style, soft, whimsical images
- **GPU-Accelerated on Colab:** Runs efficiently using an NVIDIA A100 GPU
- **Customizable Parameters:** Adjust transformation strength, guidance scale, and inference steps for different results
- **Negative Prompts for Artifact Reduction:** Enhances image clarity by avoiding blurriness, oversaturation, and unwanted artifacts

## Setup & Usage

### 1. Clone Repository & Install Dependencies
```bash
!pip install --upgrade diffusers transformers accelerate torch torchvision torchaudio xformers
```

### 2. Upload Your Input Image
Ensure your input image (`my_photo.jpg`) is placed in the `input_images` folder.

### 3. Run the Script
```python
!python advanced_art_style.py
```

## Parameters
| Parameter              | Description                                        | Default Value |
|------------------------|----------------------------------------------------|---------------|
| `chosen_style`        | Image style (default: Ghibli)                     | `ghibli`      |
| `prompt_text`         | Description of the desired image output           | Studio Ghibli Masterpiece |
| `input_file`          | Input image file name                             | `my_photo.jpg` |
| `output_file`         | Output image file name                            | `styled_image.png` |
| `transformation_strength` | Strength of transformation (0.0 - 1.0)       | `0.3`         |
| `guidance`            | Scale of prompt adherence (higher = stronger)     | `10`          |
| `inference_steps`     | Number of refinement steps (higher = detailed)    | `500`         |

## Example Output
| Input Image                      | Ghibli-Style Output |
|----------------------------------|--------------------|
| ![input](input_images/input.jpg) | ![output](output_images/styled_image.png) |

## Notes
- Higher `guidance` values increase prompt adherence but may reduce originality.
- A100 GPU is recommended for smooth execution due to high VRAM requirements.
- Negative prompts (`blurry, low quality, dark, oversaturated`) prevent unwanted artifacts.

## License
This project is open-source and free to use .

---
Enjoy generating your Ghibli-style images! 🚀



### Local Installation
Clone the repository and install the required dependencies:
bash
'''
git clone https://github.com/yourusername/advanced-art-style.git
cd advanced-art-style
pip install diffusers transformers accelerate torch torchvision torchaudio xformers matplotlib pillow --upgrade
'''
## Installation

### Requirements
- Python 3.8 or higher
- [PyTorch](https://pytorch.org/)
- [Diffusers](https://github.com/huggingface/diffusers)
- [Transformers](https://github.com/huggingface/transformers)
- [Accelerate](https://github.com/huggingface/accelerate)
- [xformers](https://github.com/facebookresearch/xformers)
- Matplotlib
- Pillow

### Local Installation
Clone the repository and install the required dependencies:
 bash
    git clone https://github.com/yourusername/advanced-art-style.git
    cd advanced-art-style
    pip install diffusers transformers accelerate torch torchvision torchaudio xformers matplotlib pillow --upgrade

!git clone https://github.com/yourusername/advanced-art-style.git
%cd advanced-art-style
!pip install diffusers transformers accelerate torch torchvision torchaudio xformers matplotlib pillow --upgrade

python advanced_art_style.py --style best --prompt "A breathtaking masterpiece painting in a hyper-realistic, surreal style" --input my_photo.jpg --output styled_image.png --strength 1.0 --guidance_scale 8.0 --steps 50

from google.colab import files
uploaded = files.upload()  # Upload your image file(s)
import shutil
for filename in uploaded.keys():
    shutil.move(filename, "input_images/" + filename)
print("Uploaded and moved files to input_images/")

!python advanced_art_style.py --style best --prompt "A breathtaking masterpiece painting in a hyper-realistic, surreal style" --input my_photo.jpg --output styled_image.png --strength 1.0 --guidance_scale 8.0 --steps 50
##  Professor Genius