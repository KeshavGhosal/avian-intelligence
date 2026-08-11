"""
Standalone PyTorch Model Inference Script for Bird Identification.

Usage:
    python run_inference.py [image_path]

If no image path is provided, generates a test sample image and runs prediction.
"""

import sys
import json
import logging
from pathlib import Path
from PIL import Image

from model import BirdInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("run_inference")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if image_path and Path(image_path).exists():
        logger.info(f"Loading input image from: {image_path}")
        image = Image.open(image_path)
    else:
        logger.info("No image path provided or file not found. Generating sample test image...")
        image = Image.new("RGB", (300, 300), color=(120, 180, 90))
        sample_path = "sample_bird.jpg"
        image.save(sample_path)
        logger.info(f"Saved sample test image to '{sample_path}'")

    logger.info("Initializing PyTorch BirdInferenceEngine...")
    engine = BirdInferenceEngine(checkpoint_path="bird_model.pth")
    
    logger.info("Running forward pass & classification...")
    result = engine.predict(image)

    print("\n" + "="*50)
    print(" BIRD CLASSIFICATION RESULT ")
    print("="*50)
    print(json.dumps(result, indent=2))
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
