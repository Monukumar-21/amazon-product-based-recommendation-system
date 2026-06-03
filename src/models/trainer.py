import torch.optim as optim

class TwoTowerTrainer:
    def __init__(self, model: TwoTowerModel, lr: float = 1e-3, device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # CrossEntropyLoss applied to similarity matrix acts as InfoNCE Loss
        self.criterion = nn.CrossEntropyLoss() 

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0

        for user_ids, item_ids in dataloader:
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)

            self.optimizer.zero_grad()
            
            # scores shape: (batch_size, batch_size)
            scores = self.model(user_ids, item_ids)
            
            # The correct item for user i is item i (the diagonal of the matrix)
            # targets: [0, 1, 2, ..., batch_size - 1]
            targets = torch.arange(scores.size(0)).to(self.device)
            
            # Compute loss
            loss = self.criterion(scores, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)