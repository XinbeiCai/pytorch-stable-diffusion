import torch
import numpy as np


class DDPMSampler:
    def __init__(self, generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.0120):
        self.start_step = None
        self.num_inference_steps = None
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def set_strength(self, strength=1):
        """
            Set how much noise to add to the input image.
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep, latents, model_output):
        t = timestep
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)

        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # predict x_0 formula(15)
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** (0.5)

        # compute coefficients formula(7)
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # Compute predicted previous sample µ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (variance ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def ddim_step(self, timestep, latents, model_output, eta=0.0):
        t = timestep
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)

        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one

        sqrt_alpha_prod_t = alpha_prod_t.sqrt()
        sqrt_alpha_prod_t_prev = alpha_prod_t_prev.sqrt()
        sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t).sqrt()

        # Predict x0 from current latent and model predicted noise (formula (12))
        pred_x0 = (latents - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t

        # Compute sigma (DDIM noise scale)
        sigma = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)).sqrt()

        # Compute direction pointing to x_t
        dir_xt = (1 - alpha_prod_t_prev - sigma ** 2).sqrt() * model_output

        # Compute x_{t-1}
        pred_prev_sample = sqrt_alpha_prod_t_prev * pred_x0 + dir_xt

        # Optionally add noise
        if eta > 0 and t > 0:
            device = model_output.device
            noise = torch.randn(latents, generator=self.generator, device=device, dtype=model_output.dtype)
            pred_prev_sample += sigma * noise

        return pred_prev_sample

    def add_noise(self, original_samples, timesteps):
        alphas_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, *[1] * (original_samples.dim() - 1))

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, *[1] * (original_samples.dim() - 1))

        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device,
                            dtype=original_samples.dtype)
        noise_samples = (sqrt_alpha_prod * original_samples) + sqrt_one_minus_alpha_prod * noise
        return noise_samples, noise
