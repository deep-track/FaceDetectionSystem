import os
import logging
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

logger = logging.getLogger("deeptrack.image")

LOCAL_MODEL_PATH = "data/best_swin.pth"
HF_REPO_ID       = "dkkinyua/fakecatcher"
HF_FILENAME      = "best_swin.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SwinTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = timm.create_model(
            "swin_small_patch4_window7_224", pretrained=False, num_classes=0
        )
        num_features = self.base_model.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.base_model(x)
        output   = self.classifier(features)
        return output, features


def _resolve_model_path() -> str:
    """Use local weights if present, otherwise use HuggingFace cache."""
    if os.path.exists(LOCAL_MODEL_PATH):
        logger.info(f"Using local weights: {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH

    from huggingface_hub import hf_hub_download
    logger.info(f"Resolving from HuggingFace cache ({HF_REPO_ID})...")
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )
    logger.info(f"Resolved to: {path}")
    return path

def _load_weights(model_path: str, device: torch.device) -> SwinTransformer:
    model      = SwinTransformer(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Swin weights loaded.")
    return model


class ImagePredictor:
    def __init__(self):
        self.device = DEVICE
        model_path  = _resolve_model_path()
        self.model  = _load_weights(model_path, self.device)

    def predict(self, image: Image.Image) -> dict:
        tensor = preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output, _ = self.model(tensor)
            probs     = torch.softmax(output, dim=1)[0]
            conf, idx = torch.max(probs, 0)
        label = ["Real", "Fake"][idx.item()]
        return {
            "prediction":            label,
            "confidence_percentage": round(conf.item() * 100, 2),
            "raw_scores": {
                "Real": round(probs[0].item() * 100, 2),
                "Fake": round(probs[1].item() * 100, 2),
            },
        }