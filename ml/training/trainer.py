"""
Training infrastructure for GraphVAE.

Implements training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import json
from datetime import datetime

from ..models import HierarchicalEncoder, HybridDecoder
from ..losses import SimplifiedCompositeLoss
from ..data import CircuitDataset, collate_circuit_batch


class VAETrainer:
    """
    Trainer for GraphVAE model.

    Handles:
        - Training loop with batching
        - Validation loop
        - Checkpointing
        - Logging and metrics tracking
        - Learning rate scheduling
        - Early stopping

    Args:
        encoder: HierarchicalEncoder model
        decoder: HybridDecoder model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        checkpoint_dir: Directory for saving checkpoints
        log_interval: Log every N batches
        val_interval: Validate every N epochs
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
        val_interval: int = 1
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.val_interval = val_interval

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []

        # For gradient clipping
        self.max_grad_norm = 1.0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of average metrics for the epoch
        """
        self.encoder.train()
        self.decoder.train()

        epoch_metrics = {
            'total_loss': [],
            'recon_total': [],
            'recon_topo': [],
            'recon_edge': [],
            'topo_accuracy': [],
            'tf_total': [],
            'kl_loss': []
        }

        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            loss, metrics = self._forward_pass(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                self.max_grad_norm
            )

            self.optimizer.step()

            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(metrics[key])

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                self._log_batch(batch_idx, len(self.train_loader), metrics)

            self.global_step += 1

        # Compute average metrics
        avg_metrics = {
            key: sum(values) / len(values) if values else 0.0
            for key, values in epoch_metrics.items()
        }

        epoch_time = time.time() - epoch_start_time
        avg_metrics['epoch_time'] = epoch_time

        return avg_metrics

    def validate(self) -> Dict[str, float]:
        """
        Run validation loop.

        Returns:
            Dictionary of validation metrics
        """
        self.encoder.eval()
        self.decoder.eval()

        val_metrics = {
            'total_loss': [],
            'recon_total': [],
            'recon_topo': [],
            'topo_accuracy': [],
            'tf_total': [],
            'kl_loss': []
        }

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)

                # Forward pass
                loss, metrics = self._forward_pass(batch)

                # Accumulate metrics
                for key in val_metrics:
                    if key in metrics:
                        val_metrics[key].append(metrics[key])

        # Average metrics
        avg_metrics = {
            key: sum(values) / len(values) if values else 0.0
            for key, values in val_metrics.items()
        }

        return avg_metrics

    def train(self, num_epochs: int, early_stopping_patience: int = 20):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
        """
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print("="*70 + "\n")

        best_epoch = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Update loss function epoch for weight scheduling
            if hasattr(self.loss_fn, 'set_epoch'):
                self.loss_fn.set_epoch(epoch, num_epochs)

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Print train metrics
            self._print_metrics("Train", train_metrics)

            # Validation
            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)

                # Print val metrics
                self._print_metrics("Val", val_metrics)

                # Check for improvement
                current_val_loss = val_metrics['total_loss']

                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    best_epoch = epoch
                    patience_counter = 0

                    # Save best model
                    self.save_checkpoint('best.pt', is_best=True)
                    print(f"✅ New best model! Val loss: {current_val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"⚠️  No improvement ({patience_counter}/{early_stopping_patience})")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning rate: {current_lr:.6f}")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                print(f"Best val loss: {self.best_val_loss:.4f} at epoch {best_epoch + 1}")
                break

        # Training complete
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best val loss: {self.best_val_loss:.4f} at epoch {best_epoch + 1}")
        print(f"Total epochs: {self.current_epoch + 1}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print("="*70 + "\n")

        # Save final checkpoint
        self.save_checkpoint('final.pt')

        # Save training history
        self.save_history()

    def _forward_pass(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through encoder and decoder.

        Args:
            batch: Batch from dataloader

        Returns:
            loss: Total loss
            metrics: Dictionary of metrics
        """
        # Encode
        z, mu, logvar = self.encoder(
            batch['graph'].x,
            batch['graph'].edge_index,
            batch['graph'].edge_attr,
            batch['graph'].batch,
            batch['poles'],
            batch['zeros']
        )

        # Decode
        decoder_output = self.decoder(z, hard=False)

        # Compute edge batch
        edge_batch = batch['graph'].batch[batch['graph'].edge_index[0]]

        # Compute loss
        encoder_output = (z, mu, logvar)
        loss, metrics = self.loss_fn(
            encoder_output,
            decoder_output,
            batch['filter_type'],
            batch['graph'].edge_attr,
            edge_batch,
            batch['poles'],
            batch['zeros']
        )

        return loss, metrics

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        # Move graph data
        batch['graph'] = batch['graph'].to(self.device)

        # Move filter type
        batch['filter_type'] = batch['filter_type'].to(self.device)

        # Move poles/zeros (lists of tensors)
        batch['poles'] = [p.to(self.device) for p in batch['poles']]
        batch['zeros'] = [z.to(self.device) for z in batch['zeros']]

        # Move gain and freq_response
        batch['gain'] = batch['gain'].to(self.device)
        batch['freq_response'] = batch['freq_response'].to(self.device)

        return batch

    def _log_batch(self, batch_idx: int, total_batches: int, metrics: Dict):
        """Log batch metrics."""
        progress = (batch_idx + 1) / total_batches * 100
        print(
            f"  [{batch_idx + 1}/{total_batches}] ({progress:.1f}%) "
            f"Loss: {metrics.get('total_loss', 0):.4f} | "
            f"Recon: {metrics.get('recon_total', 0):.4f} | "
            f"TF: {metrics.get('tf_total', 0):.2e} | "
            f"KL: {metrics.get('kl_loss', 0):.4f} | "
            f"Acc: {metrics.get('topo_accuracy', 0):.2%}"
        )

    def _print_metrics(self, prefix: str, metrics: Dict):
        """Print summary metrics."""
        print(f"\n{prefix} Metrics:")
        print(f"  Total Loss:    {metrics.get('total_loss', 0):.4f}")
        print(f"  Reconstruction: {metrics.get('recon_total', 0):.4f}")
        print(f"    - Topology:   {metrics.get('recon_topo', 0):.4f}")
        print(f"    - Edge:       {metrics.get('recon_edge', 0):.4f}")
        print(f"  Transfer Func: {metrics.get('tf_total', 0):.2e}")
        print(f"  KL Divergence: {metrics.get('kl_loss', 0):.4f}")
        print(f"  Topo Accuracy: {metrics.get('topo_accuracy', 0):.2%}")

        if 'epoch_time' in metrics:
            print(f"  Epoch Time:    {metrics['epoch_time']:.1f}s")

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            print(f"💾 Saved best checkpoint: {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"📂 Loaded checkpoint: {checkpoint_path}")
        print(f"   Epoch: {self.current_epoch}, Best val loss: {self.best_val_loss:.4f}")

    def save_history(self):
        """Save training history to JSON."""
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1
        }

        history_path = self.checkpoint_dir / 'training_history.json'

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"📊 Saved training history: {history_path}")
