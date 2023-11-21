import torch
from Dataloader import Dataloader, label_map
from model import Autoencodermodel
from sklearn.metrics import f1_score, accuracy_score
import os

from train import class_criterion

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(__file__)

# Load the saved model from the same directory as test.py
model_save_path = os.path.join(script_dir, 'trained_model.pth')
model = Autoencodermodel(num_classes=len(label_map))
model.load_state_dict(torch.load(model_save_path))
model = model.to(device)
model.eval()

test_dataset = Dataloader(split='test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)


# Define a function for evaluation
def evaluate_model(model, dataloader):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for feat, _, label, _ in dataloader:
            feat = feat.float().to(device)
            label = label.long().to(device)

            _, class_pred = model(feat)
            _, predicted = torch.max(class_pred.data, 1)

            # Collect true and predicted labels for accuracy and F1 score calculation
            y_true.extend(label.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate accuracy and F1 score using library functions
    accuracy = accuracy_score(y_true, y_pred)
    average_test_loss = class_criterion(class_pred, label)
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy * 100, average_test_loss, f1


test_accuracy, test_loss, test_f1 = evaluate_model(model, test_dataloader)

results_file = os.path.join(script_dir, 'test_results.txt')
with open(results_file, 'w') as f:
    f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.6f}\n")
    f.write(f"F1 Score: {test_f1:.6f}\n")

print(f"Test results saved to {results_file}")
