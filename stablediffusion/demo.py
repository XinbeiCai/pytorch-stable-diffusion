import os
import argparse
import model_loader
import pipeline
from PIL import Image, UnidentifiedImageError
from transformers import CLIPTokenizer
import torch

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    elif v.lower() in ("false", "0", "no"):
        return False
    else:
        raise argparse.ArgumentTypeError("Mode must be True or False")

def load_tokenizer(vocab_path, merges_path):
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError("Tokenizer files not found.")
    return CLIPTokenizer(vocab_path, merges_file=merges_path)

def load_image(image_path):
    if not image_path or not os.path.exists(image_path):
        print(f"[WARN] Image path not found or None: {image_path}")
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        print(f"[ERROR] Cannot open image: {image_path}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Text/Image-to-Image Generation Pipeline")

    parser.add_argument("--mode", type=str2bool, default=True, help="True = image-to-image, False = text-to-image")
    parser.add_argument("--prompt", type=str, default="A dog is running on the grass", help="Text prompt for generation")
    parser.add_argument("--image_path", type=str, default="../images/dog.jpg", help="Input image path (only used if mode=True)")
    parser.add_argument("--uncond_prompt", type=str, default="", help="Unconditional prompt")

    parser.add_argument("--model_path", type=str, default="../data/v1-5-pruned-emaonly.ckpt")
    parser.add_argument("--vocab_path", type=str, default="../data/vocab.json")
    parser.add_argument("--merges_path", type=str, default="../data/merges.txt")
    parser.add_argument("--output_path", type=str, default="output.png")

    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="ddpm",help= "ddim")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Mode: {'image-to-image' if args.mode else 'text-to-image'}")
    print(f"Prompt: {args.prompt}")

    # Load tokenizer
    try:
        tokenizer = load_tokenizer(args.vocab_path, args.merges_path)
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}")
        return

    # Load model
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file not found: {args.model_path}")
        return
    try:
        models = model_loader.preload_models_from_standard_weights(args.model_path, device=device)
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return

    # Image or not
    input_image = None


    if args.mode:  # image-to-image
        input_image = load_image(args.image_path)
        if input_image is None:
            print("[ERROR] --mode True but image could not be loaded.")
            return
    strength = args.strength

    try:
        output_image = pipeline.generate(
            prompt=args.prompt,
            uncond_prompt=args.uncond_prompt,
            input_image=input_image,
            strength=strength,
            do_cfg=True,
            cfg_scale=args.cfg_scale,
            sampler_name=args.sampler,
            n_inference_steps=args.steps,
            seed=args.seed,
            models=models,
            device=device,
            idle_device="cpu",
            tokenizer=tokenizer
        )
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return

    try:
        img = Image.fromarray(output_image)
        img.save(args.output_path)
        print(f"Saved to {args.output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save image: {e}")

if __name__ == "__main__":
    main()
