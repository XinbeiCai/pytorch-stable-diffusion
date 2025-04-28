# Stable Diffusion Demo

This repository provides a demo for both **Text-to-Image** and **Image-to-Image** generation using a Stable Diffusion model.

---

## 1. Download Weights and Tokenizer Files

Please download the following files from [this Gitee repo](https://gitee.com/hf-models/stable-diffusion-v1-5) and place them into the `data/` folder:

- `v1-5-pruned-emaonly.ckpt`
- `vocab.json`
- `merges.txt`

---

## 2. Installation Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```



## 3. Sampling

### Text-to-Image Generation

Run the following command:

```bash
python demo.py --mode False --prompt "A dog is running on the grass"
```

output：

<img src=".\output\text2dog.png" alt="output" style="zoom:50%;" />

### Image-to-Image Generation

**Example 1**

```bash
python demo.py --mode True -- prompt "A dog is running on the grass" --image_path "../images/mountains.jpg" 
```

input:

<img src=".\images\dog.jpg" alt="output" style="zoom:100%;" />

output:

<img src=".\output\image2dog.png" alt="output" style="zoom:50%;" />



**Example2:**

```bash
python demo.py --mode True -- prompt "A fantasy landscape, trending on artstation" --image_path "../images/mountains.jpg" 
```

input image：

<img src=".\images\mountains.jpg" alt="output" style="zoom:50%;" />

output image：

<img src=".\output\image2mountain.png" alt="output" style="zoom:50%;" />



## 4. finetune with textual inversion

train：

```
python train_textual_inversion.py
```

test：

```
python textual_demo.py
```



## 5. finetune with lora

train:

```
python train_lora.py
```

test:

```
python lora_demo.py
```



## 6. Special Thanks

This project was inspired by and based on the great work of the following repositories:

1. [hkproj/pytorch-stable-diffusion](https://github.com/hkproj/pytorch-stable-diffusion)
2. [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
3. [divamgupta/stable-diffusion-tensorflow](https://github.com/divamgupta/stable-diffusion-tensorflow)
4. [kjsman/stable-diffusion-pytorch](https://github.com/kjsman/stable-diffusion-pytorch)
5. [huggingface/diffusers](https://github.com/huggingface/diffusers)