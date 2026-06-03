import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Set

logger = logging.getLogger(__name__)

# --- Data Loading for NCF ---
class NCFDataset(Dataset):
    """
    PyTorch Dataset for Neural Collaborative Filtering.
    Expects interaction data with user, item, and optionally rating.
    Handles positive and negative sampling.
    """
    def __init__(self, df: pd.DataFrame, user_to_idx: Dict[str, int], item_to_idx: Dict[str, int],
                 num_items: int, num_neg_samples: int = 4):
        self.user_ids = df['reviewerID'].values
        self.item_ids = df['asin'].values
        self.ratings = df['overall'].values if 'overall' in df.columns else None
        
        self.user_indices = np.array([user_to_idx[uid] for uid in self.user_ids])
        self.item_indices = np.array([item_to_idx[iid] for iid in self.item_ids])
        
        self.num_items = num_items
        self.num_neg_samples = num_neg_samples
        
        # Cache user's positive interactions for negative sampling
        self._user_positive_items = {}
        for uid, iid in zip(self.user_indices, self.item_indices):
            if uid not in self._user_positive_items:
                self._user_positive_items[uid] = set()
            self._user_positive_items[uid].add(iid)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Tuple[int, int, float, List[int]]:
        user_idx = self.user_indices[idx]
        item_idx = self.item_indices[idx]
        
        # Rating is typically 1 for implicit feedback (if not provided)
        # For explicit, we can scale ratings to [0, 1]
        rating = float(self.ratings[idx]) if self.ratings is not None else 1.0
        if self.ratings is not None:
            rating = (rating - 1.0) / 4.0 # Scale to [0, 1] for prediction

        # Negative Sampling (Crucial for NCF training)
        neg_items = []
        user_pos_items = self._user_positive_items.get(user_idx, set())
        
        while len(neg_items) < self.num_neg_samples:
            neg_item = np.random.randint(self.num_items)
            if neg_item not in user_pos_items:
                neg_items.append(neg_item)
                
        return user_idx, item_idx, rating, neg_items

# --- NCF Model Definition ---
class NCF(nn.Module):
    """
    Neural Collaborative Filtering Model. Combines GMF and MLP paths.
    """
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 32, 
                 mlp_hidden_dims: List[int] = [64, 32, 16], dropout_rate: float = 0.2):
        super(NCF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout_rate = dropout_rate

        # --- Embeddings ---
        # GMF Embeddings
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)
        
        # MLP Embeddings
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

        # --- MLP Layers ---
        mlp_layers = []
        input_dim = embedding_dim * 2 # User + Item embeddings
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim)) # Batch Normalization for stability
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout_rate))
            input_dim = hidden_dim
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        # Final output layer
        # It needs to combine GMF output and MLP output
        # GMF output is embedding_dim (element-wise product)
        # MLP output is the last hidden_dim
        self.output_layer = nn.Linear(embedding_dim + mlp_hidden_dims[-1], 1)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for embeddings and dense layers."""
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        nn.init.kaiming_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        # GMF Path
        user_gmf = self.user_embedding_gmf(user_indices)
        item_gmf = self.item_embedding_gmf(item_indices)
        gmf_output = torch.mul(user_gmf, item_gmf) # Element-wise product

        # MLP Path
        user_mlp = self.user_embedding_mlp(user_indices)
        item_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=1)
        mlp_output = self.mlp_layers(mlp_input)
        
        # Concatenate GMF and MLP outputs
        final_input = torch.cat([gmf_output, mlp_output], dim=1)
        
        # Predict probability (output layer)
        prediction = self.output_layer(final_input)
        return torch.sigmoid(prediction) # Sigmoid for probability output

# --- NCF Trainer ---
class NCFTrainer:
    """
    Trainer class for the NCF model.
    Manages data loading, training loop, and evaluation.
    """
    def __init__(self, model: NCF, lr: float = 1e-3, weight_decay: float = 1e-4, 
                 device: str = 'cpu', early_stopping_patience: int = 5):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.BCELoss() # Binary Cross-Entropy Loss for probability prediction
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 100) -> None:
        logger.info(f"Starting NCF Training on device: {self.device}")
        
        for epoch in range(num_epochs):
            self.model.train() # Set model to training mode
            total_loss = 0.0
            
            for batch_idx, (user_indices, item_indices, ratings, neg_item_indices) in enumerate(train_loader):
                # Move data to device
                user_indices = user_indices.to(self.device)
                item_indices = item_indices.to(self.device)
                ratings = ratings.to(self.device).float().unsqueeze(1) # Target for positive item
                
                # Create positive and negative samples for training
                # Positive sample
                predictions_pos = self.model(user_indices, item_indices)
                loss_pos = self.criterion(predictions_pos, ratings)
                
                # Negative samples
                neg_item_indices = torch.cat(neg_item_indices, dim=1).to(self.device) # Flatten neg items
                user_indices_neg = user_indices.unsqueeze(1).repeat(1, neg_item_indices.size(1)).view(-1)
                item_indices_neg = neg_item_indices.view(-1)
                
                predictions_neg = self.model(user_indices_neg, item_indices_neg)
                # Target for negative items is 0.0
                loss_neg = self.criterion(predictions_neg, torch.zeros_like(predictions_neg))

                # Total loss for the batch: positive loss + average negative loss
                loss = loss_pos + loss_neg / len(neg_item_indices[0]) # Normalize by number of neg samples
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            
            # --- Validation ---
            val_loss = self.evaluate(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # --- Early Stopping ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # In production, save the best model:
                # torch.save(self.model.state_dict(), 'best_ncf_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluates the model on the validation set."""
        self.model.eval() # Set model to evaluation mode
        total_val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation
            for batch_idx, (user_indices, item_indices, ratings, neg_item_indices) in enumerate(val_loader):
                user_indices = user_indices.to(self.device)
                item_indices = item_indices.to(self.device)
                ratings = ratings.to(self.device).float().unsqueeze(1)
                
                # Only predict for positive items during evaluation
                predictions = self.model(user_indices, item_indices)
                loss = self.criterion(predictions, ratings)
                total_val_loss += loss.item()
        return total_val_loss / len(val_loader)

    def get_recommendations(self, user_id: str, all_users_idx: Dict[str, int], all_items_idx: Dict[str, int], 
                            user_history_items: Set[str], k: int = 10) -> List[str]:
        """Generates top-K recommendations for a given user."""
        if user_id not in all_users_idx:
            logger.warning(f"User ID '{user_id}' not found in model's user index.")
            return []

        self.model.eval()
        user_idx = all_users_idx[user_id]
        
        user_idx_tensor = torch.tensor([user_idx], device=self.device)
        
        # Prepare all item indices for prediction
        all_item_indices = torch.arange(len(all_items_idx), device=self.device)
        
        # Predict scores for all items
        with torch.no_grad():
            item_scores = self.model(user_idx_tensor.repeat(len(all_item_indices)), all_item_indices).squeeze()
        
        # Convert scores to a NumPy array for easier manipulation
        item_scores_np = item_scores.cpu().numpy()
        
        # Get top K item indices and filter out items user has interacted with
        recommended_items = []
        # Sort items by score in descending order
        sorted_item_indices = np.argsort(item_scores_np)[::-1]
        
        for item_idx in sorted_item_indices:
            item_id = list(all_items_idx.keys())[list(all_items_idx.values()).index(item_idx)] # Map index back to item ID
            if item_id not in user_history_items:
                recommended_items.append(item_id)
            if len(recommended_items) == k:
                break
        
        return recommended_items----