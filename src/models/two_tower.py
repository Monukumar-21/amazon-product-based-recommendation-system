import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class UserTower(nn.Module):
    """
    Encodes User IDs and User Features into a dense embedding.
    """
    def __init__(self, num_users: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # In a real system, you would concatenate user age, location, etc. here
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        x = self.user_embedding(user_ids)
        x = self.fc(x)
        # Crucial: Normalize embeddings so dot product equals cosine similarity
        return F.normalize(x, p=2, dim=1)

class ItemTower(nn.Module):
    """
    Encodes Item IDs and Item Features into a dense embedding.
    """
    def __init__(self, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # In a real system, you'd add item category, brand, price here
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self.item_embedding(item_ids)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

class TwoTowerModel(nn.Module):
    """
    The full Two-Tower Retrieval Model.
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(num_items, embedding_dim)
        # Temperature parameter controls the sharpness of the softmax
        self.temperature = nn.Parameter(torch.tensor(0.1))

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculates similarity scores between users and items.
        """
        user_embs = self.user_tower(user_ids)
        item_embs = self.item_tower(item_ids)
        
        # Calculate dot product similarity
        # user_embs shape: (batch_size, embedding_dim)
        # item_embs shape: (batch_size, embedding_dim)
        # scores shape: (batch_size, batch_size) -> In-batch negative interaction matrix
        scores = torch.matmul(user_embs, item_embs.T) / self.temperature
        return scores