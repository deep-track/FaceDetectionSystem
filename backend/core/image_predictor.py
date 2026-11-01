import os
import logging
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from backend.core.model_files import resolve_model_file, LANDMARKER_URL

logger = logging.getLogger("deeptrack.image")

MODEL_FILENAME = "best_swin.pth"
ARCH           = "swin_small_patch4_window7_224"
CLASSES        = ["Real", "Fake"]      # index 0 = Real, index 1 = Fake (training uses the same order)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SwinTransformer(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        """pretrained=True loads ImageNet weights for the backbone (used when training)."""
        super().__init__()
        self.base_model = timm.create_model(
            ARCH, pretrained=pretrained, num_classes=0
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


def _load_weights(model_path: str, device: torch.device):
    """Returns (model, threshold). The threshold is calibrated by training/train_swin.py;
    older checkpoints without one use 0.5."""
    model      = SwinTransformer(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    threshold  = 0.5
    if isinstance(checkpoint, dict) and "threshold" in checkpoint:
        threshold = float(checkpoint["threshold"])

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Swin weights loaded (decision threshold P(fake) >= {threshold:.3f}).")
    return model, threshold


def _make_cropper():
    """
    Crop to the face before classifying, exactly as the training data was prepared
    (scripts/prepare_image_dataset.py). Set FACE_CROP=0 to classify the whole image.
    """
    if os.getenv("FACE_CROP", "1") == "0":
        logger.info("Face cropping disabled (FACE_CROP=0).")
        return None
    try:
        from backend.core.face_crop import FaceCropper
        return FaceCropper(resolve_model_file("face_landmarker.task", LANDMARKER_URL))
    except Exception as e:
        logger.warning(f"Face cropping unavailable, classifying whole images: {e}")
        return None


class ImagePredictor:
    def __init__(self):
        self.device  = DEVICE
        # raises FileNotFoundError with instructions if best_swin.pth has not been trained/added yet
        model_path   = resolve_model_file(MODEL_FILENAME)
        self.model, self.threshold = _load_weights(model_path, self.device)
        override = os.getenv("IMAGE_THRESHOLD", "").strip()
        if override:
            t = float(override)
            if not 0.0 < t < 1.0:
                raise ValueError(f"IMAGE_THRESHOLD must be between 0 and 1, got {override}")
            logger.info(f"IMAGE_THRESHOLD={t} overrides the cut-off stored in the model ({self.threshold:.3f}).")
            self.threshold = t
        self.cropper = _make_cropper()

    def predict(self, image: Image.Image) -> dict:
        face_detected = None
        if self.cropper is not None:
            image, face_detected = self.cropper.crop(image)

        tensor = preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output, _ = self.model(tensor)
            probs     = torch.softmax(output, dim=1)[0]
        p_fake, t = probs[1].item(), self.threshold
        is_fake   = p_fake >= t
        # how far the score is past the cut-off: 50 = right at it, 100 = certain.
        # With t = 0.5 this equals the winning class probability (the old behaviour).
        conf = 50 + 50 * ((p_fake - t) / (1 - t) if is_fake else (t - p_fake) / t)
        return {
            "prediction":            CLASSES[1 if is_fake else 0],
            "confidence_percentage": round(conf, 2),
            "raw_scores": {
                "Real": round(probs[0].item() * 100, 2),
                "Fake": round(probs[1].item() * 100, 2),
            },
            "decision_threshold":    round(t, 4),
            "face_detected":         face_detected,
        }

    def close(self):
        if self.cropper is not None:
            self.cropper.close()
