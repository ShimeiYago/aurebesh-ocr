"""
Train CRNN recognizer with MobileNetV3-Small backbone for Aurebesh text recognition.
"""

import os
# Enable MPS fallback for operations not yet supported on MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
import numpy as np
from PIL import Image
from tqdm import tqdm
import lmdb
from doctr.models.recognition import crnn_mobilenet_v3_small
from torchvision import transforms
import cv2

from utils import setup_logger, ensure_dir, load_config, get_run_id, get_charset


class AurebeshRecognitionDataset(Dataset):
    """Dataset for Aurebesh text recognition using LMDB."""
    
    def __init__(self, lmdb_path: Path, charset: str, transform=None):
        self.lmdb_path = Path(lmdb_path)
        self.charset = charset
        self.transform = transform
        
        # Don't open LMDB in __init__ to avoid pickling issues
        self.env = None
        
        # Get all keys from LMDB
        env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, map_size=50 * 1024 * 1024 * 1024)
        with env.begin() as txn:
            cursor = txn.cursor()
            self.keys = [key.decode() for key, _ in cursor]
            self.length = len(self.keys)
        env.close()
        
        # Create char to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Open LMDB if not already open (happens in each worker process)
        if self.env is None:
            self.env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, map_size=50 * 1024 * 1024 * 1024)
        
        with self.env.begin() as txn:
            # Get data from LMDB using the stored key
            key = self.keys[idx].encode()
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {self.keys[idx]} not found in LMDB")
            data = pickle.loads(value)
            
            # Load image
            img_bytes = data['image']
            img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Get text
            text = data['text']
            
            # Apply transforms
            if self.transform:
                # Convert to PIL for torchvision transforms
                pil_image = Image.fromarray(img_array)
                img_tensor = self.transform(pil_image)
            else:
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            
            # Convert text to indices
            text_indices = [self.char_to_idx.get(char, 0) for char in text]  # 0 for unknown chars
            text_tensor = torch.tensor(text_indices, dtype=torch.long)
            
            return img_tensor, text_tensor, text


class RecognizerTrainer:
    def __init__(
        self,
        config_path: Path,
        data_dir: Path,
        output_dir: Path,
        device: str = 'mps',
        resume_from: Optional[Path] = None
    ):
        self.config = load_config(config_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        
        # For MPS, we need to handle CTC loss on CPU
        self.ctc_device = torch.device('cpu') if device == 'mps' else self.device
        
        # Load charset
        charset_path = Path(__file__).parent.parent / "configs" / self.config['charset']
        self.charset = get_charset(charset_path)
        self.vocab_size = len(self.charset)
        self.blank_idx = self.vocab_size  # CTC blank token should be after all vocab
        
        # Setup directories
        self.run_id = get_run_id('rec')
        self.checkpoint_dir = ensure_dir(self.output_dir / 'weights' / 'rec')
        self.log_dir = ensure_dir(self.output_dir / 'log' / self.run_id)
        
        # Setup logger
        self.logger = setup_logger('train_rec', self.log_dir, use_tensorboard=True)
        
        # Setup model
        self.model = self._setup_model()
        
        # Setup data
        self.train_loader, self.val_loader = self._setup_data()
        
        # Setup training
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = nn.CTCLoss(blank=self.blank_idx, zero_infinity=True)
        
        # Training state
        self.start_epoch = 0
        self.best_metric = float('inf')  # Lower CER is better
        
        if resume_from:
            self._load_checkpoint(resume_from)
    
    def _setup_model(self) -> nn.Module:
        """Setup CRNN model with MobileNetV3 backbone."""
        # Load pretrained model
        model = crnn_mobilenet_v3_small(pretrained=True, vocab=self.charset)
        
        # Adapt output layer for our vocab size + blank token
        output_size = self.vocab_size + 1  # +1 for CTC blank
        if hasattr(model, 'classifier') and model.classifier[-1].out_features != output_size:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, output_size)
        
        model = model.to(self.device)
        
        self.logger.info(f"Loaded CRNN with MobileNetV3-Small backbone, vocab_size={self.vocab_size}")
        return model
    
    def _setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """Setup data loaders."""
        # Transforms
        transform = transforms.Compose([
            transforms.Resize((32, 128)),  # Standard size for text recognition
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Datasets
        train_dataset = AurebeshRecognitionDataset(
            self.data_dir / 'train' / 'lmdb',
            charset=self.charset,
            transform=transform
        )
        val_dataset = AurebeshRecognitionDataset(
            self.data_dir / 'val' / 'lmdb',
            charset=self.charset,
            transform=transform
        )
        
        # Data loaders with custom collate function
        def collate_fn(batch):
            images, texts, text_strs = zip(*batch)
            
            # Stack images
            images = torch.stack(images, 0)
            
            # Handle variable length texts
            text_lengths = [len(t) for t in texts]
            max_len = max(text_lengths)
            
            # Pad texts
            padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long)
            for i, text in enumerate(texts):
                padded_texts[i, :len(text)] = text
            
            return images, padded_texts, torch.tensor(text_lengths), text_strs
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False,
            collate_fn=collate_fn
        )
        
        self.logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        return train_loader, val_loader
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        opt_config = self.config['optimizer']
        
        if opt_config['name'] == 'AdamW':
            optimizer = AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['name']}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        sched_config = self.config['scheduler']
        
        if sched_config['name'] == 'OneCycleLR':
            steps_per_epoch = len(self.train_loader)
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=sched_config['max_lr'],
                epochs=self.config['epochs'],
                steps_per_epoch=steps_per_epoch,
                pct_start=sched_config['pct_start']
            )
        elif sched_config['name'] == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config.get('eta_min', 0)
            )
        elif sched_config['name'] == 'StepLR':
            scheduler = StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_config['name']}")
        
        return scheduler
    
    def _calculate_cer(self, preds: List[str], targets: List[str]) -> float:
        """Calculate Character Error Rate using Levenshtein distance."""
        total_chars = 0
        total_errors = 0
        
        for pred, target in zip(preds, targets):
            total_chars += len(target)
            # Use Levenshtein distance for accurate CER
            errors = self._levenshtein_distance(pred, target)
            total_errors += errors
        
        return total_errors / max(total_chars, 1)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _decode_predictions(self, outputs: torch.Tensor) -> List[str]:
        """Decode CTC outputs to text."""
        # Get predictions
        _, preds = outputs.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        
        # Decode batch
        decoded = []
        batch_size = outputs.size(1)
        
        for i in range(batch_size):
            # Get sequence for this sample
            seq = preds[i::batch_size]
            
            # Remove blanks and repeated characters
            chars = []
            prev = None
            for idx in seq:
                idx_val = idx.item() if hasattr(idx, 'item') else idx
                # Use correct blank index (vocab_size, not 0)
                if idx_val != self.blank_idx and idx_val != prev:
                    if 0 <= idx_val < len(self.charset):
                        chars.append(self.charset[idx_val])
                prev = idx_val
            
            decoded.append(''.join(chars))
        
        return decoded
    
    def _train_epoch(self, epoch: int):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        for batch_idx, (images, texts, text_lengths, text_strs) in enumerate(pbar):
            images = images.to(self.device)
            texts = texts.to(self.device)
            text_lengths = text_lengths.to(self.device)
            
            # Forward pass - CRNN requires target during training
            # For MPS, we need to disable built-in loss calculation and do it on CPU
            if self.device.type == 'mps':
                # Get logits only without loss calculation
                # Pass dummy targets to avoid ValueError but ignore the built-in loss
                outputs = self.model(images, target=[''] * images.size(0), return_model_output=True)
                
                if isinstance(outputs, dict) and 'out_map' in outputs:
                    logits = outputs['out_map']
                else:
                    raise ValueError("Unexpected model output format")
                
                # Calculate CTC loss on CPU
                # CTC loss expects log_probs in shape [T, N, C] (time, batch, classes)
                # But model outputs [N, T, C] (batch, time, classes), so we need to transpose
                log_probs = logits.transpose(0, 1).log_softmax(2).cpu()
                texts_cpu = texts.cpu()
                
                # Input lengths should be the sequence length (time dimension) for each sample
                # All sequences have the same length after transforms
                input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long, device='cpu')
                text_lengths_cpu = text_lengths.cpu()
                
                loss = self.criterion(log_probs, texts_cpu, input_lengths, text_lengths_cpu)
                # Move loss back to original device for backward pass
                loss = loss.to(self.device)
            else:
                # Normal path for non-MPS devices
                outputs = self.model(images, target=list(text_strs), return_model_output=True)
                
                if isinstance(outputs, dict):
                    logits = outputs['out_map']
                    # The model might already provide a loss
                    if 'loss' in outputs and outputs['loss'] is not None:
                        loss = outputs['loss']
                    else:
                        # Calculate CTC loss manually
                        # CTC loss expects log_probs in shape [T, N, C]
                        log_probs = logits.transpose(0, 1).log_softmax(2)
                        input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long)
                        loss = self.criterion(log_probs, texts, input_lengths, text_lengths)
                else:
                    # Fallback for direct tensor output
                    logits = outputs
                    # CTC loss expects log_probs in shape [T, N, C]
                    log_probs = logits.transpose(0, 1).log_softmax(2)
                    input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long)
                    loss = self.criterion(log_probs, texts, input_lengths, text_lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Only step OneCycleLR per batch
            if self.config['scheduler']['name'] == 'OneCycleLR':
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Decode predictions for CER
            preds = self._decode_predictions(logits)
            all_preds.extend(preds)
            all_targets.extend(text_strs)
            
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Log to tensorboard
            if hasattr(self.logger, 'tensorboard_writer'):
                global_step = epoch * len(self.train_loader) + batch_idx
                self.logger.tensorboard_writer.add_scalar('train/loss', loss.item(), global_step)
                self.logger.tensorboard_writer.add_scalar('train/lr', 
                                                         self.optimizer.param_groups[0]['lr'], 
                                                         global_step)
        
        # Calculate CER
        cer = self._calculate_cer(all_preds, all_targets)
        return avg_loss, cer
    
    def _validate(self, epoch: int):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch_idx, (images, texts, text_lengths, text_strs) in enumerate(pbar):
                images = images.to(self.device)
                texts = texts.to(self.device)
                text_lengths = text_lengths.to(self.device)
                
                # Forward pass - handle MPS the same way as training
                if self.device.type == 'mps':
                    outputs = self.model(images, target=[''] * images.size(0), return_model_output=True)
                else:
                    outputs = self.model(images, target=list(text_strs), return_model_output=True)
                
                # Extract logits and calculate loss
                if isinstance(outputs, dict) and 'out_map' in outputs:
                    logits = outputs['out_map']
                else:
                    raise ValueError("Unexpected model output format")
                
                # Calculate CTC loss
                if self.device.type == 'mps':
                    # Calculate CTC loss on CPU for MPS
                    log_probs = logits.transpose(0, 1).log_softmax(2).cpu()
                    texts_cpu = texts.cpu()
                    input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long, device='cpu')
                    text_lengths_cpu = text_lengths.cpu()
                    loss = self.criterion(log_probs, texts_cpu, input_lengths, text_lengths_cpu)
                else:
                    # Normal CTC loss calculation
                    log_probs = logits.transpose(0, 1).log_softmax(2)
                    input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long)
                    loss = self.criterion(log_probs, texts, input_lengths, text_lengths)
                
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                # Decode predictions
                preds = self._decode_predictions(logits)
                all_preds.extend(preds)
                all_targets.extend(text_strs)
                
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Calculate CER
        cer = self._calculate_cer(all_preds, all_targets)
        
        # Log to tensorboard
        if hasattr(self.logger, 'tensorboard_writer'):
            self.logger.tensorboard_writer.add_scalar('val/loss', avg_loss, epoch)
            self.logger.tensorboard_writer.add_scalar('val/cer', cer, epoch)
            
            # Log sample predictions
            for i in range(min(5, len(all_preds))):
                self.logger.tensorboard_writer.add_text(
                    f'val/sample_{i}',
                    f"Target: {all_targets[i]}\nPred: {all_preds[i]}",
                    epoch
                )
        
        return avg_loss, cer
    
    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metric': metric,
            'config': self.config,
            'charset': self.charset
        }
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pt'
        torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with CER: {metric:.4f}")
        
        # Always save best.pt on first epoch for smoke test
        if epoch == 0 and not (self.checkpoint_dir / 'best.pt').exists():
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            self.logger.info("Saved initial best.pt for smoke test")
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['metric']
        
        self.logger.info(f"Resumed from epoch {checkpoint['epoch']} with CER {checkpoint['metric']:.4f}")
    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config['epochs']} epochs")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Train
            train_loss, train_cer = self._train_epoch(epoch)
            
            # Validate
            val_loss, val_cer = self._validate(epoch)
            
            # Save checkpoint
            is_best = val_cer < self.best_metric
            if is_best:
                self.best_metric = val_cer
            
            self._save_checkpoint(epoch, val_cer, is_best)
            
            # Log
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_cer={train_cer:.4f}, "
                f"val_loss={val_loss:.4f}, val_cer={val_cer:.4f}, best_cer={self.best_metric:.4f}"
            )
            
            # Step schedulers that update per epoch
            if self.config['scheduler']['name'] in ['CosineAnnealingLR', 'StepLR']:
                self.scheduler.step()
        
        self.logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Aurebesh text recognizer")
    parser.add_argument("--config", type=Path, default="configs/train_rec.yaml", help="Training config")
    parser.add_argument("--data_dir", type=Path, default="data/synth", help="Dataset directory")
    parser.add_argument("--output_dir", type=Path, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override epochs if specified
    if args.epochs:
        config['epochs'] = args.epochs
    
    # Create trainer
    trainer = RecognizerTrainer(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        resume_from=args.resume
    )
    
    # Override config after initialization
    trainer.config = config
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()