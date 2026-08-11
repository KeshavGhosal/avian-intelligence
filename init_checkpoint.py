"""
Helper Utility Script to Initialize a Sample Checkpoint File (bird_model.pth).

Creates a valid state dictionary file for the BirdClassifier PyTorch model
to facilitate immediate testing and verification of the FastAPI server out-of-the-box.
"""

import sys
import torch
import logging

from model import BirdClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("init_checkpoint")


def create_initial_checkpoint(output_path: str = "bird_model.pth"):
    logger.info("Instantiating BirdClassifier architecture...")
    model = BirdClassifier(num_classes=200, pretrained=True)
    
    logger.info(f"Saving baseline state_dict to '{output_path}'...")
    torch.save({"state_dict": model.state_dict()}, output_path)
    logger.info(f"Successfully generated initial PyTorch checkpoint: '{output_path}'")


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "bird_model.pth"
    create_initial_checkpoint(out_file)
