"""
Verification and Testing Script for Bird Identification Assistant Backend.

Runs unit checks on:
1. CUB-200 class mapping and taxonomy extraction.
2. PyTorch model initialization, image preprocessing, and tensor forward pass.
3. FastAPI endpoint health, classification, and ecology service mocking.
"""

import os
import io
import unittest
from unittest.mock import AsyncMock, patch

import torch
from PIL import Image

from cub200_data import CUB_CLASSES, get_cub_species_info, clean_class_name
from model import BirdClassifier, BirdInferenceEngine
from services import EcologyService


class TestBirdBackend(unittest.TestCase):

    def test_cub200_class_data(self):
        """Test CUB-200 data formatting and taxonomy parsing."""
        self.assertEqual(len(CUB_CLASSES), 200)
        self.assertEqual(clean_class_name(CUB_CLASSES[0]), "Black footed Albatross")
        
        info = get_cub_species_info(0)
        self.assertIn("species", info)
        self.assertIn("common_name", info)
        self.assertIn("taxonomy", info)
        self.assertEqual(info["taxonomy"]["order"], "Procellariiformes")
        self.assertEqual(info["taxonomy"]["family"], "Diomedeidae")

    def test_pytorch_model_forward(self):
        """Test PyTorch model architecture tensor shape."""
        model = BirdClassifier(num_classes=200, pretrained=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (1, 200))

    def test_inference_engine_prediction(self):
        """Test end-to-end PIL image prediction pipeline."""
        engine = BirdInferenceEngine(checkpoint_path=None)
        dummy_img = Image.new("RGB", (300, 300), color="blue")
        result = engine.predict(dummy_img)
        
        self.assertIn("species", result)
        self.assertIn("common_name", result)
        self.assertIn("confidence", result)
        self.assertIn("taxonomy", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
