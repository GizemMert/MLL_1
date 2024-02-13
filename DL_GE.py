from torch.utils.data import Dataset
import torch

class GeneExpressionDataset(Dataset):
    def __init__(self, expressions, labels):
        self.expressions = expressions
        self.labels = labels

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        expression = self.expressions[idx]
        label = self.labels[idx]
        return expression, label


# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(numeric_labels, dtype=torch.long)

# Initialize dataset
dataset = GeneExpressionDataset(X_tensor, y_tensor)
