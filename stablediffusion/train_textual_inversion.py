import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from stablediffusion.textual_inversion_dataset import TextualInversionDataset
from stablediffusion.model_loader import preload_models_from_standard_weights
from ddpm import DDPMSampler
from transformers import CLIPTokenizer

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# ==== 配置 ====
device = torch.device("cuda")
placeholder_token = "<cat-toy>"
initializer_token = "cat"
model_path = "../data/v1-5-pruned-emaonly.ckpt"
save_path = "../data/learned_embedding.pt"
max_train_steps = 1000

# ==== 加载模型 ====
models = preload_models_from_standard_weights(model_path, device)
clip = models["clip"]
encoder = models["encoder"]
diffusion = models["diffusion"]

# ==== 加载 tokenizer ====
tokenizer = CLIPTokenizer(vocab_file="../data/vocab.json", merges_file="../data/merges.txt")
if placeholder_token not in tokenizer.get_vocab():
    num_added = tokenizer.add_tokens([placeholder_token])
    if num_added > 0:
        # 手动扩展embedding
        old_emb = clip.embedding.token_embedding
        old_num, emb_dim = old_emb.weight.shape
        new_num = len(tokenizer)

        # 新建更大的embedding层，拷贝旧权重
        new_emb = nn.Embedding(new_num, emb_dim)
        new_emb.weight.data[:old_num] = old_emb.weight.data

        # 把clip里的embedding换成新的
        clip.embedding.token_embedding = new_emb.to(device)

placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
initializer_token_id = tokenizer.convert_tokens_to_ids(initializer_token)

# ==== 准备可学习的 embedding ====
embedding_layer = clip.embedding.token_embedding
init_embed = embedding_layer.weight.data[initializer_token_id].clone()
learned_embed = nn.Parameter(init_embed.clone().detach().to(device))

# 冻结 clip 其他参数
for param in clip.parameters():
    param.requires_grad = False

optimizer = optim.Adam([learned_embed], lr=1e-3)

# ==== 加载数据 ====
dataset = TextualInversionDataset("../images/cat_toy", placeholder_token=placeholder_token)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ==== 时间步 embedding函数 ====
def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

# ==== to_idle 工具函数 ====
def to_idle(module, device=torch.device("cpu")):
    return module.to(device)

# ==== 训练主循环 ====
generator = torch.Generator(device=device).manual_seed(42)
sampler = DDPMSampler(generator=generator)

for step in tqdm(range(max_train_steps)):
    loss_sum = 0
    for image, prompt in dataloader:
        image = image.to(device)

        # 编码 prompt
        input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77).input_ids.to(device)
        # lookup 原embedding
        input_embeds = embedding_layer(input_ids)

        # 替换 placeholder token 位置
        mask = (input_ids == placeholder_token_id).unsqueeze(-1)  # (b, l, 1)
        input_embeds = input_embeds * (~mask) + learned_embed.unsqueeze(0).unsqueeze(0) * mask
        input_embeds += clip.embedding.position_embedding
        # 过 clip 得到 context
        context = clip(input_embeds=input_embeds)

        # image 编码到 latent
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
        latents = encoder(image, noise=encoder_noise)

        # 随机采样时间步
        t = torch.randint(0, 1000, (1,), generator=generator, device=device).long()
        t_val = t.item()

        # 给 latent 加噪声
        noisy_latents, noise = sampler.add_noise(latents, torch.tensor(t_val))
        # timestep embedding
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

# ==== 释放模型内存 ====
to_idle(clip)
to_idle(encoder)
to_idle(diffusion)

# ==== 保存 embedding ====
torch.save({placeholder_token: learned_embed.detach().cpu()}, save_path)
print(f"Saved learned embedding to {save_path}")
