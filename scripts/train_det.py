"""
Train DBNet detector with MobileNetV3-Large backbone for Aurebesh text detection.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import numpy as np
from PIL import Image
from tqdm import tqdm
from doctr.models.detection import db_mobilenet_v3_large
from torchvision import transforms
import cv2

# Install shapely if not available: pip install shapely
try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not available. Polygon shrinking will be disabled.")

from utils import setup_logger, ensure_dir, load_config, get_run_id


class AurebeshDetectionDataset(Dataset):
    """Dataset for Aurebesh text detection."""
    
    def __init__(self, data_dir: Path, split: str = 'train', transform=None, shrink_ratio: float = 0.6):
        self.data_dir = Path(data_dir) / split
        self.images_dir = self.data_dir / 'images'
        self.transform = transform
        self.shrink_ratio = shrink_ratio  # For better text instance separation
        
        # Load annotations
        with open(self.data_dir / 'annotations.json', 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image ID to annotations mapping
        self.image_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_anns:
                self.image_to_anns[img_id] = []
            self.image_to_anns[img_id].append(ann)
        
        # Filter images that have annotations
        self.images = [img for img in self.coco_data['images'] 
                      if img['id'] in self.image_to_anns]
    
    def _shrink_polygon(self, polygon: np.ndarray, shrink_ratio: float) -> np.ndarray:
        """Shrink polygon to create better text instance separation."""
        if not SHAPELY_AVAILABLE:
            # Fallback: simple erosion using cv2
            return self._shrink_polygon_cv2(polygon, shrink_ratio)
            
        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
            
            # Create shapely polygon
            poly = Polygon(polygon)
            if not poly.is_valid:
                # Try to fix invalid polygon
                poly = poly.buffer(0)
            
            # Calculate shrink distance based on polygon area
            area = poly.area
            shrink_distance = shrink_ratio * np.sqrt(area) / 2
            
            # Shrink polygon (negative buffer)
            shrunken = poly.buffer(-shrink_distance)
            
            if shrunken.is_empty or shrunken.area < 1:
                # If shrinking makes polygon too small, return original with minimal shrink
                shrunken = poly.buffer(-1)
                
            if shrunken.is_empty:
                return polygon  # Return original if still empty
            
            # Extract coordinates
            if hasattr(shrunken, 'exterior'):
                coords = np.array(shrunken.exterior.coords[:-1])  # Remove duplicate last point
            else:
                return polygon  # Return original if shrinking failed
                
            return coords
            
        except Exception:
            # If any error occurs, return original polygon
            return polygon
    
    def _shrink_polygon_cv2(self, polygon: np.ndarray, shrink_ratio: float) -> np.ndarray:
        """Fallback polygon shrinking using OpenCV erosion."""
        try:
            # Create a mask from polygon
            bbox = cv2.boundingRect(polygon.astype(np.int32))
            x, y, w, h = bbox
            
            # Create mask slightly larger than bounding box
            mask_size = (h + 20, w + 20)
            mask = np.zeros(mask_size, dtype=np.uint8)
            
            # Translate polygon to mask coordinate system
            translated_polygon = polygon.copy()
            translated_polygon[:, 0] -= (x - 10)
            translated_polygon[:, 1] -= (y - 10)
            
            # Fill polygon in mask
            cv2.fillPoly(mask, [translated_polygon.astype(np.int32)], 255)
            
            # Calculate erosion kernel size based on polygon area
            area = cv2.contourArea(polygon.astype(np.int32))
            kernel_size = max(1, int(shrink_ratio * np.sqrt(area) / 4))
            
            # Erode mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            eroded_mask = cv2.erode(mask, kernel, iterations=1)
            
            # Find contours in eroded mask
            contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return polygon  # Return original if erosion removed everything
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Convert back to original coordinate system
            result_polygon = largest_contour.squeeze()
            if len(result_polygon.shape) == 1:
                return polygon  # Return original if contour is invalid
                
            result_polygon[:, 0] += (x - 10)
            result_polygon[:, 1] += (y - 10)
            
            return result_polygon.astype(np.float32)
            
        except Exception:
            return polygon
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.images_dir / img_info['file_name']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        img_array = np.array(image)
        
        # Get annotations
        anns = self.image_to_anns[img_info['id']]
        
        # Convert to target format for DBNet
        h, w = img_array.shape[:2]
        
        # Create segmentation map
        seg_map = np.zeros((h, w), dtype=np.float32)
        
        for ann in anns:
            # Get polygon points
            segmentation = ann['segmentation'][0]  # First (and only) polygon
            points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
            
            # Apply shrink to improve text instance separation
            shrunken_points = self._shrink_polygon(points.astype(np.float32), self.shrink_ratio)
            shrunken_points = shrunken_points.astype(np.int32)
            
            # Fill polygon with shrunken version
            cv2.fillPoly(seg_map, [shrunken_points], 1.0)
        
        # Apply transforms
        if self.transform:
            # Convert to PIL for torchvision transforms
            pil_image = Image.fromarray(img_array)
            image = self.transform(pil_image)
            # Resize seg_map to match transformed image size
            seg_map = cv2.resize(seg_map, (image.shape[2], image.shape[1]))
        else:
            image = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        
        target = {
            'maps': torch.from_numpy(seg_map).unsqueeze(0),  # Add channel dimension
            'polygons': [ann['segmentation'][0] for ann in anns],
            'texts': [ann['text'] for ann in anns]
        }
        
        return image, target


class DetectorTrainer:
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
        
        # Setup directories
        self.run_id = get_run_id('det')
        self.checkpoint_dir = ensure_dir(self.output_dir / 'weights' / 'det')
        self.log_dir = ensure_dir(self.output_dir / 'log' / self.run_id)
        
        # Setup logger
        self.logger = setup_logger('train_det', self.log_dir, use_tensorboard=True)
        
        # Setup model
        self.model = self._setup_model()
        
        # Setup data
        self.train_loader, self.val_loader = self._setup_data()
        
        # Setup training
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        # AMP is not fully supported on MPS yet, disable it
        self.scaler = None if self.device.type == 'mps' else (
            GradScaler('cpu') if self.config.get('amp', True) and self.device.type == 'cpu' else (
                GradScaler(self.device.type) if self.config.get('amp', True) else None
            )
        )
        
        # Training state
        self.start_epoch = 0
        self.best_metric = 0
        
        if resume_from:
            self._load_checkpoint(resume_from)
    
    def _setup_model(self) -> nn.Module:
        """Setup DBNet model with MobileNetV3 backbone."""
        # Load pretrained model
        model = db_mobilenet_v3_large(pretrained=True)
        
        # Create a wrapper to extract probability maps during training
        class DBNetTrainingWrapper(nn.Module):
            def __init__(self, dbnet_model):
                super().__init__()
                self.dbnet = dbnet_model
                
            def forward(self, x):
                # Use the model's forward with return_model_output=True
                # This returns the raw model outputs including probability maps
                outputs = self.dbnet(x, target=None, return_model_output=True)
                
                # The model returns a dict with 'out_map' which contains the probability map
                if isinstance(outputs, dict) and 'out_map' in outputs:
                    return outputs['out_map']
                else:
                    # Fallback: try to get features manually
                    feat_maps = self.dbnet.feat_extractor(x)
                    feat_concat = self.dbnet.fpn(feat_maps)
                    logits = self.dbnet.prob_head(feat_concat)
                    return torch.sigmoid(logits)
        
        # Wrap the model for training
        model = DBNetTrainingWrapper(model)
        model = model.to(self.device)
        
        self.logger.info(f"Loaded DBNet with MobileNetV3-Large backbone")
        return model
    
    def _setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """Setup data loaders."""
        # Transforms
        transform = transforms.Compose([
            transforms.Resize((self.config.get('input_size', 1024), self.config.get('input_size', 1024))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get shrink ratio from config (default 0.6 for better separation)
        shrink_ratio = self.config.get('gt_generation', {}).get('shrink_ratio', 0.6)
        self.logger.info(f"Using shrink_ratio: {shrink_ratio} for GT generation")
        
        # Datasets
        train_dataset = AurebeshDetectionDataset(
            self.data_dir, split='train', transform=transform, shrink_ratio=shrink_ratio
        )
        val_dataset = AurebeshDetectionDataset(
            self.data_dir, split='val', transform=transform, shrink_ratio=shrink_ratio
        )
        
        # Custom collate function to handle variable number of annotations
        def detection_collate(batch):
            images = []
            targets = []
            
            for sample in batch:
                image, target = sample
                images.append(image)
                targets.append(target)
            
            # Stack images into batch
            images = torch.stack(images, 0)
            
            # Keep targets as list since they have variable sizes
            return images, targets
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False,
            collate_fn=detection_collate
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False,
            collate_fn=detection_collate
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
        
        if sched_config['name'] == 'CosineAnnealingLR':
            # Account for warmup
            warmup_epochs = sched_config.get('warmup_epochs', 0)
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'] - warmup_epochs,
                eta_min=sched_config['eta_min']
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_config['name']}")
        
        return scheduler
    
    def _dice_loss(self, pred_maps: torch.Tensor, target_maps: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss for segmentation."""
        smooth = 1e-6
        
        pred_flat = pred_maps.view(-1)
        target_flat = target_maps.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def _train_epoch(self, epoch: int):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # Stack target maps from list of targets
            target_maps = []
            for target in targets:
                target_maps.append(target['maps'])
            target_maps = torch.stack(target_maps, 0).to(self.device)
            
            # Forward pass
            with autocast(device_type=self.device.type, enabled=self.scaler is not None):
                pred_maps = self.model(images)  # Now directly returns probability maps
                loss = self._dice_loss(pred_maps, target_maps)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += float(loss.item())
            avg_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Log to tensorboard
            if hasattr(self.logger, 'tensorboard_writer'):
                global_step = epoch * len(self.train_loader) + batch_idx
                self.logger.tensorboard_writer.add_scalar('train/loss', loss.item(), global_step)
        
        return avg_loss
    
    def _validate(self, epoch: int):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(self.device)
                
                # Stack target maps from list of targets
                target_maps = []
                for target in targets:
                    target_maps.append(target['maps'])
                target_maps = torch.stack(target_maps, 0).to(self.device)
                
                # Forward pass
                with autocast(device_type=self.device.type, enabled=self.scaler is not None):
                    pred_maps = self.model(images)  # Now directly returns probability maps
                    loss = self._dice_loss(pred_maps, target_maps)
                
                total_loss += float(loss.item())
                avg_loss = total_loss / (batch_idx + 1)
                
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Log to tensorboard
        if hasattr(self.logger, 'tensorboard_writer'):
            self.logger.tensorboard_writer.add_scalar('val/loss', avg_loss, epoch)
        
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool = False):
        """Save model checkpoint."""
        # Save the actual DBNet model state, not the wrapper
        model_state = self.model.dbnet.state_dict() if hasattr(self.model, 'dbnet') else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metric': metric,
            'config': self.config
        }
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pt'
        torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with metric: {metric:.4f}")
        
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
        
        self.logger.info(f"Resumed from epoch {checkpoint['epoch']} with metric {checkpoint['metric']:.4f}")
    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config['epochs']} epochs")
        
        # Warmup
        warmup_epochs = self.config['scheduler'].get('warmup_epochs', 0)
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Warmup learning rate
            if epoch < warmup_epochs:
                warmup_lr = self.config['optimizer']['lr'] * (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Train
            train_loss = self._train_epoch(epoch)
            
            # Validate
            val_loss = self._validate(epoch)
            
            # Update scheduler (after warmup)
            if epoch >= warmup_epochs:
                self.scheduler.step()
            
            # Save checkpoint
            metric = 1 - val_loss  # Convert loss to metric (higher is better)
            is_best = metric > self.best_metric
            if is_best:
                self.best_metric = metric
            
            self._save_checkpoint(epoch, metric, is_best)
            
            # Log
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"metric={metric:.4f}, best={self.best_metric:.4f}"
            )
        
        self.logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Aurebesh text detector")
    parser.add_argument("--config", type=Path, default="configs/train_det.yaml", help="Training config")
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
    
    # Create trainer with modified config
    class ConfigOverride:
        def __init__(self, config_dict):
            self.config = config_dict
    
    config_obj = ConfigOverride(config)
    
    # Modify DetectorTrainer to accept config dict
    trainer = DetectorTrainer(
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