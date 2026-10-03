import os
import torch

from torch import nn
from typing import Sequence

checkpoint_dir = "/resources/seminar-04-unet/checkpoints"
model_dir = "/resources/seminar-04-unet/models"
id_dir = "/resources/seminar-04-unet/ids"


def save(run_name: str, epoch: int, valid_metric_history: Sequence,
         generator, gen_optimizer, gen_scheduler,
         discriminator, dis_optimizer, dis_scheduler):
    checkpoint = {
        "epoch": epoch,
        "valid_metric_history": valid_metric_history,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "gen_optimizer": gen_optimizer.state_dict(),
        "dis_optimizer": dis_optimizer.state_dict(),
        "gen_scheduler": None if gen_scheduler is None
                              else gen_scheduler.state_dict(),
        "dis_scheduler": None if dis_scheduler is None
                              else dis_scheduler.state_dict()
    }
    checkpoint_path = f"{checkpoint_dir}/{run_name}.chk"
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load(run_name: str,
         generator, gen_optimizer, gen_scheduler,
         discriminator, dis_optimizer, dis_scheduler):
    checkpoint = read(run_name)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
    dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
    if gen_scheduler is not None:
        gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])
    if dis_scheduler is not None:
        dis_scheduler.load_state_dict(checkpoint['dis_scheduler'])
    return checkpoint["epoch"], checkpoint["valid_metric_history"]


def read(run_name: str):
    return torch.load(f"{checkpoint_dir}/{run_name}.chk")


def save_model(run_name: str, epoch: int, model: nn.Module):
    model_path = f"{model_dir}/{run_name}_epoch{epoch}.pth"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)


def load_model(filename: str, model: nn.Module):
    model_path = f"{model_dir}/{filename}.pth"
    model.load_state_dict(torch.load(model_path))


def load_pretrained(filename: str, key: str, model: nn.Module):
    try:
        load_model(filename, model)
    except FileNotFoundError:
        checkpoint = read(filename)
        model.load_state_dict(checkpoint[key])


def store_id(run_name: str, id: str):
    id_path = f"{id_dir}/{run_name}.txt"
    os.makedirs(id_dir, exist_ok=True)
    with open(id_path, "w") as f:
        f.write(id + "\n")


def get_id(run_name: str) -> str:
    id_path = f"{id_dir}/{run_name}.txt"
    with open(id_path, "r") as f:
        id = f.readline()[:-1]
    return id
