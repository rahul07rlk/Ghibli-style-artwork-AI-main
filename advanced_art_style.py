import os
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import matplotlib.pyplot as plt

# Use current working directory (e.g., Colab)
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use the SDXL Refiner model
MODEL_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"


# Dummy safety checker to disable content filtering
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)


def load_pipeline():
    """Load the SDXL Refiner pipeline fully on GPU."""
    print(f"Loading SDXL Refiner model from {MODEL_ID} onto GPU...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # Half precision for efficiency
        use_safetensors=True,  # Faster & safer loading
        low_cpu_mem_usage=False  # Keep model on GPU
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()  # Optimize VRAM usage
    pipe.safety_checker = dummy_safety_checker
    print(f"Pipeline loaded on device: {pipe.device}")
    return pipe


def calculate_target_size(original_width, original_height, max_dim=1024, multiple=8):
    """Calculate dimensions preserving aspect ratio."""
    aspect_ratio = original_width / original_height
    if original_width > original_height:
        new_width = max_dim
        new_height = int(max_dim / aspect_ratio)
    else:
        new_height = max_dim
        new_width = int(max_dim * aspect_ratio)
    new_width = multiple * round(new_width / multiple)
    new_height = multiple * round(new_height / multiple)
    return new_width, new_height


def generate_img2img(pipe, prompt, input_filename, output_filename, strength, guidance_scale, steps):
    """Generate Ghibli-style image using the SDXL Refiner."""
    input_path = os.path.join(INPUT_DIR, input_filename)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image '{input_path}' not found.")

    # Load and display input image
    init_image = Image.open(input_path).convert("RGB")
    print("Displaying input image:")
    plt.imshow(init_image)
    plt.axis("off")
    plt.show()

    original_width, original_height = init_image.size
    new_width, new_height = calculate_target_size(original_width, original_height, max_dim=1024)
    init_image_resized = init_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"Resized input image to {new_width}x{new_height}")

    print(f"Applying style transfer with prompt: '{prompt}'")
    try:
        output_image = pipe(
            prompt=prompt,
            image=init_image_resized,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            negative_prompt="blurry, low quality, dark, oversaturated"  # Avoid artifacts
        ).images[0]
    except Exception as e:
        print(f"Error during generation: {e}")
        output_image = Image.new('RGB', (new_width, new_height), color='black')

    output_image.save(output_path)
    print(f"Output image saved to {output_path}")
    plt.imshow(output_image)
    plt.axis("off")
    plt.show()


# --- Optimized Parameters for Trending Ghibli Style ---
chosen_style = "ghibli"
prompt_text = (
    "A detailed Studio Ghibli-style masterpiece 4k, featuring lush landscapes, soft pastel colors, intricate whimsical details, "
    "gentle magical lighting, vibrant yet harmonious tones, trending in modern anime art"
)
input_file = "my_photo.jpg"  # Ensure this is in input_images/
output_file = "styled_image.png"
transformation_strength = 0.3  # Balanced transformation for detail and input retention
guidance = 10  # Strong prompt adherence for Ghibli aesthetics
inference_steps = 500  # Sufficient steps for refined output

# --- Main Execution ---
pipe = load_pipeline()
generate_img2img(pipe, prompt_text, input_file, output_file, transformation_strength, guidance, inference_steps)