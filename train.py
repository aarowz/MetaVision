"""
Training script for MetaVision ViT model.

Handles the complete training pipeline including:
- Data loading with augmentation
- Model training and validation
- Checkpointing and early stopping
- TensorBoard logging
"""

import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random

# Add src to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model
from src.data_loader import MetasurfaceDataset, create_train_val_test_splits
from src.augmentation import get_augmentation_transform


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_name: str):
    """Get the appropriate device."""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_loss_function(loss_name: str, loss_params: dict = None):
    """Get loss function from config."""
    loss_params = loss_params or {}
    
    if loss_name.lower() == "mse":
        return nn.MSELoss()
    elif loss_name.lower() == "mae":
        return nn.L1Loss()
    elif loss_name.lower() == "huber":
        delta = loss_params.get("delta", 1.0)
        return nn.HuberLoss(delta=delta)
    elif loss_name.lower() == "combined":
        mse_weight = loss_params.get("mse_weight", 0.7)
        mae_weight = loss_params.get("mae_weight", 0.3)
        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        
        def combined_loss(pred, target):
            return mse_weight * mse_loss(pred, target) + mae_weight * mae_loss(pred, target)
        
        return combined_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def get_optimizer(model, optimizer_name: str, lr: float, weight_decay: float, optimizer_params: dict = None):
    """Get optimizer from config."""
    optimizer_params = optimizer_params or {}
    
    if optimizer_name.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_params.get("betas", [0.9, 0.999]),
            eps=optimizer_params.get("eps", 1e-8)
        )
    elif optimizer_name.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_params.get("betas", [0.9, 0.999]),
            eps=optimizer_params.get("eps", 1e-8)
        )
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=optimizer_params.get("momentum", 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name: str, scheduler_params: dict = None):
    """Get learning rate scheduler from config."""
    scheduler_params = scheduler_params or {}
    
    if scheduler_name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get("T_max", 100),
            eta_min=scheduler_params.get("eta_min", 0)
        )
    elif scheduler_name.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get("step_size", 30),
            gamma=scheduler_params.get("gamma", 0.1)
        )
    elif scheduler_name.lower() == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_params.get("factor", 0.1),
            patience=scheduler_params.get("patience", 10),
            min_lr=scheduler_params.get("min_lr", 0)
        )
    elif scheduler_name.lower() == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_epoch(model, dataloader, criterion, optimizer, device, 
                gradient_clip: float = 0.0, use_amp: bool = False, 
                aug_transform=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Apply augmentation during training
        if aug_transform is not None:
            # Augmentation expects [C, H, W] or [B, C, H, W]
            B = inputs.shape[0]
            for i in range(B):
                inputs[i] = aug_transform(inputs[i])
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir: Path, 
                   is_best: bool = False, save_top_k: int = 3):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # Save latest checkpoint
    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    torch.save(checkpoint, latest_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved best model (loss: {loss:.6f}) to {best_path}")
    
    # Save periodic checkpoint
    epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, epoch_path)
    
    # Keep only top K checkpoints (simple implementation - keep best and latest)
    # For full top-k, would need to track multiple best losses


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: Path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    return epoch, loss


def main():
    """Main training function."""
    # Load config
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup reproducibility
    seed = config["system"]["seed"]
    deterministic = config["system"]["deterministic"]
    set_seed(seed, deterministic)
    
    # Setup device
    device = get_device(config["system"]["device"])
    print(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = PROJECT_ROOT / config["system"]["checkpoint_dir"]
    log_dir = PROJECT_ROOT / config["system"]["log_dir"]
    tensorboard_dir = PROJECT_ROOT / config["system"]["tensorboard_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if config["logging"]["use_tensorboard"] else None
    
    # Create data splits
    train_idx, val_idx, test_idx = create_train_val_test_splits(
        total_files=config["data"]["total_files"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config["data"]["split_seed"]
    )
    
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")
    
    # Create datasets
    data_dir = PROJECT_ROOT / config["data"]["data_dir"]
    train_dataset = MetasurfaceDataset(
        data_dir=data_dir,
        file_indices=train_idx,
        split="train",
        normalize_input=config["data"]["normalize_input"],
        normalize_output=config["data"]["normalize_output"],
        input_norm_params=config["data"]["input_norm_params"],
        output_norm_params=config["data"]["output_norm_params"]
    )
    
    val_dataset = MetasurfaceDataset(
        data_dir=data_dir,
        file_indices=val_idx,
        split="val",
        normalize_input=config["data"]["normalize_input"],
        normalize_output=config["data"]["normalize_output"],
        input_norm_params=config["data"]["input_norm_params"],
        output_norm_params=config["data"]["output_norm_params"]
    )
    
    # Get augmentation transform
    aug_transform = get_augmentation_transform(config["data_augmentation"]) if config["data_augmentation"]["enabled"] else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["system"]["num_workers"],
        pin_memory=config["system"]["pin_memory"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["system"]["num_workers"],
        pin_memory=config["system"]["pin_memory"]
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup loss, optimizer, scheduler
    criterion = get_loss_function(
        config["training"]["loss"],
        config["training"]["loss_params"]
    )
    
    optimizer = get_optimizer(
        model,
        config["training"]["optimizer"],
        config["training"]["learning_rate"],
        config["training"]["weight_decay"],
        config["training"]["optimizer_params"]
    )
    
    scheduler = get_scheduler(
        optimizer,
        config["training"]["scheduler"],
        config["training"]["scheduler_params"]
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    resume_path = config["system"]["resume_from_checkpoint"]
    if resume_path is not None:
        resume_path = PROJECT_ROOT / resume_path
        if resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, resume_path, device)
            start_epoch += 1
    
    # Training loop
    num_epochs = config["training"]["num_epochs"]
    use_amp = config["training"]["mixed_precision"]
    gradient_clip = config["training"]["gradient_clip"]
    
    # Early stopping
    early_stopping = config["training"]["early_stopping"]
    patience = early_stopping["patience"] if early_stopping["enabled"] else None
    min_delta = early_stopping.get("min_delta", 0.0)
    patience_counter = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Early stopping: {'enabled' if early_stopping['enabled'] else 'disabled'}")
    if early_stopping["enabled"]:
        print(f"Patience: {patience}, Min delta: {min_delta}")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip=gradient_clip,
            use_amp=use_amp,
            aug_transform=aug_transform
        )
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Learning_Rate", current_lr, epoch)
        
        # Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        save_every_n = config["system"]["save_every_n_epochs"]
        if is_best or (epoch + 1) % save_every_n == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                checkpoint_dir, is_best=is_best,
                save_top_k=config["training"]["save_top_k"]
            )
        
        # Early stopping
        if early_stopping["enabled"] and patience is not None:
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()

