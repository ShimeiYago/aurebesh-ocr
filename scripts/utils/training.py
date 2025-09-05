# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


class EarlyStopper:
    """Early stopping utility to prevent overfitting during training.
    
    Args:
        patience (int): Number of epochs to wait before stopping if no improvement
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        """Check if training should be stopped early.
        
        Args:
            validation_loss (float): Current validation loss
            
        Returns:
            bool: True if training should be stopped, False otherwise
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
