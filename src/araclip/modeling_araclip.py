import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from huggingface_hub import PyTorchModelHubMixin
from transformers import BertConfig, BertModel, AutoTokenizer
from open_clip import (
    create_model,
)


class MultilingualClipEdited(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, transformer_cfg, in_features, out_features, tokenizer_repo_id_or_path
    ):
        super().__init__()
        self.transformer = BertModel(BertConfig(**transformer_cfg))
        self.clip_head = nn.Linear(in_features=in_features, out_features=out_features)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_repo_id_or_path,
        )

    def forward(self, txt):
        txt_tok = self.tokenizer(txt, padding=True, return_tensors="pt")
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok["attention_mask"]
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.clip_head(embs)


class AraClip(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="araclip",
    repo_url="https://github.com/Arabic-Clip/Araclip",
    tags=["clip"],
):
    def __init__(
        self,
        transformer_cfg,
        in_features,
        out_features,
        tokenizer_repo_id_or_path="Arabic-Clip/bert-base-arabertv2-ViT-B-16-SigLIP-512-epoch-155-trained-2M",
    ):
        super().__init__()
        self.text_model = MultilingualClipEdited(
            transformer_cfg,
            in_features,
            out_features,
            tokenizer_repo_id_or_path,
        )

        self.clip_model = create_model("ViT-B-16-SigLIP-512", pretrained_hf=False)
        self.compose = transforms.Compose(
            [
                transforms.Resize(
                    (512, 512),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
        )

    def language_model(self, queries):
        return np.asarray(self.text_model(queries).detach().to("cpu"))

    def embed(self, text: str = None, image: Image.Image = None):
        if text is None and image is None:
            raise ValueError("Please provide either text or image input")

        if text is not None and image is not None:
            text_features = self.language_model([text])[0]
            text_features = text_features / np.linalg.norm(text_features)

            img_tensor = self.compose(image).unsqueeze(0)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(img_tensor)
            image_features = image_features.squeeze(0).cpu().numpy()
            image_features = image_features / np.linalg.norm(image_features)

            return text_features, image_features

        elif text is not None:
            text_features = self.language_model([text])[0]
            return text_features / np.linalg.norm(text_features)

        else:
            img_tensor = self.compose(image).unsqueeze(0)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(img_tensor)
            image_features = image_features.squeeze(0).cpu().numpy()
            return image_features / np.linalg.norm(image_features)
