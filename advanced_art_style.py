import os
import argparse
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import matplotlib.pyplot as plt

# Define directories (using current working directory in Colab)
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Supported artistic styles and their model IDs
STYLE_MODELS = {
    "best": "stabilityai/stable-diffusion-2-1",  # High-quality model
    "ghibli": "nitrosocke/Ghibli-Diffusion",
    "anime": "hakurei/waifu-diffusion",
    "van_gogh": "stabilityai/stable-diffusion-2",
    "watercolor": "lambdalabs/sd-paint-by-example",
}


# Dummy safety checker to disable content filtering
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)


def load_img2img_pipeline(style: str):
    """Load the image-to-image pipeline onto GPU."""
    model_id = STYLE_MODELS.get(style.lower())
    if model_id is None:
        raise ValueError(f"Style '{style}' not supported. Choose from: {list(STYLE_MODELS.keys())}")
    print(f"Loading img2img model for style '{style}' from {model_id} onto GPU...")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=False
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    # We disable the safety checker for testing; you can re-enable it if needed.
    pipe.safety_checker = dummy_safety_checker
    print(f"Pipeline loaded on device: {pipe.device}")
    return pipe


def calculate_target_size(original_width, original_height, max_dim=512, multiple=8):
    """Calculate new dimensions preserving aspect ratio and ensuring multiples of 8."""
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


def generate_img2img(pipe, prompt: str, input_filename: str, output_filename: str,
                     strength: float = 1.0, guidance_scale: float = 8.0, steps: int = 50):
    """Generate a styled image using the img2img pipeline."""
    input_path = os.path.join(INPUT_DIR, input_filename)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image '{input_path}' not found.")

    init_image = Image.open(input_path).convert("RGB")
    print("Displaying input image:")
    plt.imshow(init_image)
    plt.axis("off")
    plt.show()

    original_width, original_height = init_image.size
    new_width, new_height = calculate_target_size(original_width, original_height)
    init_image_resized = init_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"Resized input image to {new_width}x{new_height}")

    print(
        f"Applying style transfer with prompt: '{prompt}', strength: {strength}, guidance_scale: {guidance_scale}, steps: {steps}")
    output_image = pipe(
        prompt=prompt,
        image=init_image_resized,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps
    ).images[0]

    output_image.save(output_path)
    print(f"Output image saved to {output_path}")
    plt.imshow(output_image)
    plt.axis("off")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Image-to-Image style transfer using GPU in Colab.")
    parser.add_argument("--style", type=str, choices=list(STYLE_MODELS.keys()), default="best",
                        help="Choose the artistic style. Options: " + ", ".join(STYLE_MODELS.keys()))
    parser.add_argument("--prompt", type=str,
                        default="A breathtaking masterpiece painting in a hyper-realistic, surreal style",
                        help="Text prompt for the desired output.")
    parser.add_argument("--input", type=str, default="my_photo.jpg",
                        help="Input image filename in input_images/ folder.")
    parser.add_argument("--output", type=str, default="styled_image.png",
                        help="Output image filename in output_images/ folder.")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Transformation strength (0.0 to 1.0).")
    parser.add_argument("--guidance_scale", type=float, default=8.0,
                        help="Guidance scale for prompt adherence.")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of inference steps.")
    args = parser.parse_args()

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipe = load_img2img_pipeline(args.style)
    generate_img2img(pipe, args.prompt, args.input, args.output, args.strength, args.guidance_scale, args.steps)


if __name__ == "__main__":
    main()
