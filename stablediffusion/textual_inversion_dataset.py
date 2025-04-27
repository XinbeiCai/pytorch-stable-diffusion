# textual_inversion_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

class TextualInversionDataset(Dataset):
    def __init__(self, image_dir, placeholder_token="<cat-toy>", image_size=512):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f)
                            for f in os.listdir(image_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        self.placeholder_token = placeholder_token
        self.templates =  [
            'a photo of a {}',
            'a rendering of a {}',
            'a cropped photo of the {}',
            'the photo of a {}',
            'a photo of a clean {}',
            'a photo of a dirty {}',
            'a dark photo of the {}',
            'a photo of my {}',
            'a photo of the cool {}',
            'a close-up photo of a {}',
            'a bright photo of the {}',
            'a cropped photo of a {}',
            'a photo of the {}',
            'a good photo of the {}',
            'a photo of one {}',
            'a close-up photo of the {}',
            'a rendition of the {}',
            'a photo of the clean {}',
            'a rendition of a {}',
            'a photo of a nice {}',
            'a good photo of a {}',
            'a photo of the nice {}',
            'a photo of the small {}',
            'a photo of the weird {}',
            'a photo of the large {}',
            'a photo of a cool {}',
            'a photo of a small {}',
        ]

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),  # Converts to [0,1]
            T.Normalize([0.5], [0.5])  # Normalizes to [-1,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text_template = random.choice(self.templates)
        prompt = text_template.format(self.placeholder_token)

        return image, prompt
