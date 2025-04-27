import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from textual_inversion_dataset import TextualInversionDataset
from model_loader import preload_models_from_standard_weights
from ddpm import DDPMSampler
from transformers import CLIPTokenizer

import math

from attention import SelfAttention, CrossAttention

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# ==== 配置 ====
device = torch.device("cuda")
model_path = "../data/v1-5-pruned-emaonly.ckpt"
save_path = "../data/lora_weights.pt"
max_train_steps = 1000
lora_rank = 4

# ==== 加载模型 ====
models = preload_models_from_standard_weights(model_path, device)
clip = models["clip"]
encoder = models["encoder"]
diffusion = models["diffusion"]  # 这里是你的 UNetModel！

# ==== 加载 tokenizer ====
tokenizer = CLIPTokenizer(vocab_file="../data/vocab.json", merges_file="../data/merges.txt")

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


apply_lora_to_unet(diffusion, rank=lora_rank, device=device)

# 只训练 LoRA 参数
lora_params = [p for p in diffusion.parameters() if p.requires_grad]
optimizer = optim.Adam(lora_params, lr=1e-4)

# ==== 加载数据 ====
dataset = TextualInversionDataset("../images/cat_toy", placeholder_token="cat_toy")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ==== 时间步 embedding函数 ====
def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

# ==== 保存 LoRA权重 ====
def save_lora_weights(unet, save_path):
    lora_modules = {}
    for name, module in unet.named_modules():
        if isinstance(module, (SelfAttention, CrossAttention)):
            for proj_name in ["in_proj", "q_proj", "k_proj", "v_proj"]:
                if hasattr(module, proj_name):
                    proj = getattr(module, proj_name)
                    if isinstance(proj, LoRALinear):
                        lora_modules[f"{name}.{proj_name}"] = proj

    # 直接把所有LoRALinear的子模块的state_dict合在一起保存
    to_save = {k: v.state_dict() for k, v in lora_modules.items()}
    torch.save(to_save, save_path)
    print(f"LoRA weights saved to {save_path}")


# ==== 训练主循环 ====
generator = torch.Generator(device=device).manual_seed(42)
sampler = DDPMSampler(generator=generator)

for step in tqdm(range(max_train_steps)):
    loss_sum = 0
    for image, prompt in dataloader:
        image = image.to(device)

        # 编码 prompt
        input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77).input_ids.to(device)
        # 过 clip 得到 context
        context = clip(tokens=input_ids)

        # image 编码到 latent
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
        latents = encoder(image, noise=encoder_noise)

        # 随机采样时间步
        t = torch.randint(0, 1000, (1,), generator=generator, device=device).long()
        t_val = t.item()

        # 给 latent 加噪声
        noisy_latents, noise = sampler.add_noise(latents, torch.tensor(t_val))
        time_emb = get_time_embedding(t_val).to(device)

        # 预测噪声
        pred_noise = diffusion(noisy_latents, context, time_emb)

        loss = F.mse_loss(pred_noise, noise)
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if step % 100 == 0:
        print(f"Step {step} - Loss: {loss_sum:.4f}")

save_lora_weights(diffusion, save_path)

