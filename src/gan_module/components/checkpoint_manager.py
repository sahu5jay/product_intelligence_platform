import os
import torch

from src.shared_utils.logger import logging


class CheckpointManager:

    def __init__(self, checkpoint_dir):

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch, generator, discriminator, optimizer_g, optimizer_d):

        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict()
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pth"
        )

        torch.save(checkpoint, checkpoint_path)

        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)

        logging.info(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path, generator, discriminator, optimizer_g, optimizer_d, device):

        checkpoint = torch.load(checkpoint_path, map_location=device)

        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

        epoch = checkpoint["epoch"]

        logging.info(f"Checkpoint loaded from {checkpoint_path}")

        return epoch