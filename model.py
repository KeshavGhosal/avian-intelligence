"""
PyTorch Model Architecture and Inference Engine for Bird Species Identification.

This module defines the PyTorch fine-tuning model (EVA-02 Base backbone for 200 CUB classes)
and the production inference engine handling 448x448 ImageNet preprocessing, tensor forward pass,
softmax confidence calculations, and species taxonomy resolution.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import timm
import torch
import torch.nn as nn
from PIL import Image

from cub200_data import CUB_CLASSES, get_cub_species_info

logger = logging.getLogger("bird_classifier.model")


class BirdClassifier(nn.Module):
    """
    PyTorch Neural Network for 200-class Bird Identification based on EVA-02 Base.
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = False):
        super(BirdClassifier, self).__init__()

        self.model = timm.create_model(
            "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting unnormalized logits for 200 species classes."""
        return self.model(x)


class BirdInferenceEngine:
    """
    Inference Manager providing image validation, timm-aligned tensor preprocessing,
    PyTorch model execution, softmax scoring, and taxonomy resolution.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = "final_bird_weights.pth",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing BirdInferenceEngine on device: {self.device}")

        self.model = BirdClassifier(num_classes=len(CUB_CLASSES), pretrained=False)

        target = Path(checkpoint_path) if checkpoint_path else None

        if target and target.exists():
            try:
                checkpoint = torch.load(target, map_location=self.device)

                if isinstance(checkpoint, dict):
                    if "state_dict" in checkpoint:
                        checkpoint = checkpoint["state_dict"]
                    elif "model_state_dict" in checkpoint:
                        checkpoint = checkpoint["model_state_dict"]

                    self.model.model.load_state_dict(checkpoint)
                elif isinstance(checkpoint, nn.Module):
                    self.model = checkpoint

                logger.info(f"Successfully loaded model weights from '{target}'")

            except Exception as e:
                logger.error(f"Failed to load state dict from '{target}': {str(e)}")
                raise e
        else:
            logger.warning(
                f"Checkpoint path '{target}' not found. Initialized default model."
            )

        self.model.to(self.device)
        self.model.eval()

        # Resolve exact transforms required by EVA-02 fine-tuned model
        data_config = timm.data.resolve_model_data_config(self.model.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image)
        return tensor.unsqueeze(0)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        try:
            tensor = self.preprocess_image(image).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_class_idx = torch.max(
                    probabilities, dim=1
                )

                class_idx = int(predicted_class_idx.item())
                confidence_score = float(confidence.item())

            species_info = get_cub_species_info(class_idx)

            return {
                "species": species_info["species"],
                "common_name": species_info["common_name"],
                "confidence": round(confidence_score, 4),
                "taxonomy": species_info["taxonomy"],
            }

        except Exception as e:
            logger.error(f"Inference error: {str(e)}", exc_info=True)
            raise RuntimeError(
                f"Failed to process image during classification: {str(e)}"
            )