import os
import argparse
import model_loader
import pipeline
from PIL import Image, UnidentifiedImageError
from transformers import CLIPTokenizer
import torch
from attention import SelfAttention, CrossAttention
import torch.nn as nn
import math

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

# ==== LoRA模块 (只针对Attention里的qkv) ====
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, scale=1.0):
        super().__init__()
        self.linear = linear_layer  # 原本的Linear层
        self.rank = rank
        self.scale = scale

        self.lora_up = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.lora_down = nn.Linear(rank, linear_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_down.weight)

    def forward(self, x):
        return self.linear(x) + self.scale * self.lora_down(self.lora_up(x))


def apply_lora_to_unet(unet, rank=4, scale=1.0, device='cuda'):
    # 先把整个unet所有参数冻结
    for param in unet.parameters():
        param.requires_grad = False

    # 再在需要的地方加LoRA
    for name, module in unet.named_modules():
        if isinstance(module, SelfAttention):
            module.in_proj = LoRALinear(module.in_proj, rank=rank, scale=scale).to(device)
        elif isinstance(module, CrossAttention):
            module.q_proj = LoRALinear(module.q_proj, rank=rank, scale=scale).to(device)
            module.k_proj = LoRALinear(module.k_proj, rank=rank, scale=scale).to(device)
            module.v_proj = LoRALinear(module.v_proj, rank=rank, scale=scale).to(device)


def load_lora_weights(unet, lora_path, device='cuda'):
    lora_state_dict = torch.load(lora_path, map_location=device)

    for name, module in unet.named_modules():
        if isinstance(module, (SelfAttention, CrossAttention)):
            for proj_name in ["in_proj", "q_proj", "k_proj", "v_proj"]:
                if hasattr(module, proj_name):
                    proj = getattr(module, proj_name)
                    if isinstance(proj, LoRALinear):
                        key = f"{name}.{proj_name}"
                        if key in lora_state_dict:
                            proj.load_state_dict(lora_state_dict[key])

    print(f"Loaded LoRA weights from {lora_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Text/Image-to-Image Generation Pipeline")

    parser.add_argument("--mode", type=str2bool, default=False, help="True = image-to-image, False = text-to-image")
    parser.add_argument("--prompt", type=str, default="a <cat-toy> is on the grass", help="Text prompt for generation")
    parser.add_argument("--image_path", type=str, default="../images/dog.jpg",
                        help="Input image path (only used if mode=True)")
    parser.add_argument("--uncond_prompt", type=str, default="", help="Unconditional prompt")

    parser.add_argument("--model_path", type=str, default="../data/v1-5-pruned-emaonly.ckpt")
    parser.add_argument("--vocab_path", type=str, default="../data/vocab.json")
    parser.add_argument("--merges_path", type=str, default="../data/merges.txt")
    parser.add_argument("--output_path", type=str, default="lora_output.png")

    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="ddpm", help="ddim or ddpm")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Mode: {'image-to-image' if args.mode else 'text-to-image'}")
    print(f"Prompt: {args.prompt}")

    # Load tokenizer

    tokenizer = load_tokenizer(args.vocab_path, args.merges_path)
    models = model_loader.preload_models_from_standard_weights(args.model_path, device=device)

    # ==== 给 UNet 打好 LoRA 结构 ====
    lora_rank = 4
    apply_lora_to_unet(models["diffusion"], rank=lora_rank, device=device)

    # ==== 加载 LoRA权重 ====
    save_path = "../data/lora_weights.pt"
    load_lora_weights(models["diffusion"], save_path, device=device)

    # Image or not
    input_image = None

    if args.mode:  # image-to-image
        input_image = load_image(args.image_path)
        if input_image is None:
            print("[ERROR] --mode True but image could not be loaded.")
            return
    strength = args.strength

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
        tokenizer=tokenizer)

    img = Image.fromarray(output_image)
    img.save(args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
