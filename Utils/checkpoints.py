import os
import glob
import re
import shutil
import torch


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def load_latest_checkpoint(checkpoint_root, model_name, fold, device):
    model_tag = sanitize_name(model_name)
    fold_dir = os.path.join(checkpoint_root, model_tag, f"fold_{fold}")
    pattern = os.path.join(fold_dir, "checkpoint_epoch_*.pt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    # Sort by epoch number (not lexicographically)
    def get_epoch_num(ckpt_path):
        match = re.search(r"checkpoint_epoch_(\d+)\.pt", ckpt_path)
        return int(match.group(1)) if match else -1

    latest_ckpt_path = max(checkpoints, key=get_epoch_num)
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    checkpoint["path"] = latest_ckpt_path
    return checkpoint


def save_checkpoint(checkpoint_root, model_name, fold, epoch, next_epoch, classifier, optimizer, interrupted=False):
    model_tag = sanitize_name(model_name)
    fold_dir = os.path.join(checkpoint_root, model_tag, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    ckpt_path = os.path.join(fold_dir, f"checkpoint_epoch_{epoch}.pt")
    checkpoint = {
        "fold": fold,
        "epoch": epoch,
        "next_epoch": next_epoch,
        "model_name": model_name,
        "model_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "interrupted": interrupted,
    }
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def cleanup_model_checkpoints(checkpoint_root, model_name):
    model_tag = sanitize_name(model_name)
    model_ckpt_dir = os.path.join(checkpoint_root, model_tag)
    if os.path.isdir(model_ckpt_dir):
        shutil.rmtree(model_ckpt_dir)
        return model_ckpt_dir
    return None


def is_fold_completed(checkpoint_dir, model_name, fold, epochs, device):
    """Check if a fold has completed all epochs."""
    latest_checkpoint = load_latest_checkpoint(checkpoint_dir, model_name, fold, device)
    if latest_checkpoint is None:
        return False
    next_epoch = int(latest_checkpoint.get("next_epoch", 0))
    return next_epoch >= epochs
