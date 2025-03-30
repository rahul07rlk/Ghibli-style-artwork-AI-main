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
- Pillow

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
