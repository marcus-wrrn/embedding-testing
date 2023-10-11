import torch
from model_parts.base_models import load_mlp_model
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import load_twitter_dataset

def evaluate_confusion_matrix(confusion_matrix: torch.Tensor):
    # Extracting values from the confusion matrix
    TN = confusion_matrix[0, 0].item()
    FP = confusion_matrix[0, 1].item()
    FN = confusion_matrix[1, 0].item()
    TP = confusion_matrix[1, 1].item()

    # Computing Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Computing Precision for the positive class
    precision = TP / (TP + FP)
    print("Confusion Matrix:")
    print(confusion_matrix)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (for positive class): {precision * 100:.2f}%")

@torch.no_grad()
def test_model(model, dataloader):
    confusion_matrix = torch.zeros(2, 2, dtype=torch.int)
    for data, target in dataloader:
        output = model(data)
        output = (output > 0.5).int()
        for t, p in zip(target.view(-1), output.view(-1)):
            confusion_matrix[int(t.item()), p.item()] += 1
    return confusion_matrix

def evaluate_classification_model(modelpath: str):
    # Loading the model
    model = load_mlp_model(modelpath)
    model.eval()
    # Loading the data
    test_data = load_twitter_dataset("./Data/twitter_sentiment/twitter.10000.test.json")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    
    confusion_matrix = test_model(model, test_dataloader)
    evaluate_confusion_matrix(confusion_matrix)

def main():
    evaluate_classification_model("./Data/twitter_sentiment/train_results/modelMLP.pth")

if __name__ == "__main__":
    main()