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


import os
import torch


def load_learned_embedding(clip_model, tokenizer, embed_path, device):
    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Embedding file not found: {embed_path}")

    # 加载保存的embedding文件
    learned_embed_dict = torch.load(embed_path, map_location=device)
    if not isinstance(learned_embed_dict, dict):
        raise ValueError(f"Expected a dict with token name as key, but got {type(learned_embed_dict)}")
    if len(learned_embed_dict) != 1:
        raise ValueError(f"Expected exactly 1 token in the embedding file, but got {len(learned_embed_dict)}")

    # 自动推断placeholder_token和learned_embed
    placeholder_token, learned_embed = list(learned_embed_dict.items())[0]

    if learned_embed.ndim == 1:
        learned_embed = learned_embed.unsqueeze(0)  # (1, dim)
    learned_embed = learned_embed.to(device)

    # 确保tokenizer包含这个token
    if placeholder_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([placeholder_token])

    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    token_embedding = clip_model.embedding.token_embedding
    old_weight = token_embedding.weight.data
    num_tokens, embed_dim = old_weight.shape

    # 如果token ID超过当前embedding尺寸，扩展embedding
    if placeholder_id >= num_tokens:
        print(f"[INFO] Extending embedding table for {placeholder_token} (id={placeholder_id})")
        new_num_tokens = placeholder_id + 1
        new_emb = torch.nn.Embedding(new_num_tokens, embed_dim).to(device)
        new_emb.weight.data[:num_tokens] = old_weight
        clip_model.embedding.token_embedding = new_emb
        token_embedding = clip_model.embedding.token_embedding  # 更新指针

    # 无论是否扩展，都写入对应位置
    with torch.no_grad():
        token_embedding.weight[placeholder_id] = learned_embed.squeeze(0)

    print(f"[INFO] Embedding for {placeholder_token} (id={placeholder_id}) injected successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Text/Image-to-Image Generation Pipeline")

    parser.add_argument("--mode", type=str2bool, default=False, help="True = image-to-image, False = text-to-image")
    parser.add_argument("--prompt", type=str, default="a <cat-toy> is on the grass", help="Text prompt for generation")
    parser.add_argument("--uncond_prompt", type=str, default="", help="Unconditional prompt")
    parser.add_argument("--image_path", type=str, default=None, help="Optional input image for img2img")

    parser.add_argument("--model_path", type=str, default="../data/v1-5-pruned-emaonly.ckpt")
    parser.add_argument("--vocab_path", type=str, default="../data/vocab.json")
    parser.add_argument("--merges_path", type=str, default="../data/merges.txt")
    parser.add_argument("--embedding_path", type=str, default="../data/learned_embedding.pt")
    parser.add_argument("--placeholder_token", type=str, default="<cat-toy>")

    parser.add_argument("--output_path", type=str, default="output.png")
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
    print(f"Prompt: {args.prompt}")

    tokenizer = load_tokenizer(args.vocab_path, args.merges_path)

    models = model_loader.preload_models_from_standard_weights(args.model_path, device=device)

    load_learned_embedding(
        clip_model=models["clip"],
        tokenizer=tokenizer,
        embed_path=args.embedding_path,
        device=device
    )

    input_image = load_image(args.image_path) if args.mode else None

    output_image = pipeline.generate(
        prompt=args.prompt,
        uncond_prompt=args.uncond_prompt,
        input_image=input_image,
        strength=args.strength,
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

    img = Image.fromarray(output_image)
    img.save(args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
