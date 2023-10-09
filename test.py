import torch
from model import MLP
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

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (for positive class): {precision * 100:.2f}%")


def evaluate_classification_model(modelpath: str, layers):
    # Loading the model
    model = MLP(layers=layers)
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    # Loading the data
    test_data = load_twitter_dataset("./Data/twitter_sentiment/twitter.10000.test.json")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    confusion_matrix = torch.zeros(2, 2, dtype=torch.int)

    with torch.no_grad():
        for data, target in test_dataloader:
            target = target
            # Process outputs
            output = model(data)
            output = (output >= 0.5).int()
            for t, p in zip(target.view(-1), output.view(-1)):
                
                confusion_matrix[int(t.item()), p.item()] += 1
    print(confusion_matrix)
    evaluate_confusion_matrix(confusion_matrix)

            
            

def main():
    evaluate_classification_model("./Data/twitter_sentiment/train_results/model_weights.pth", [768, 768 * 2, 768, 400, 768, 2])

if __name__ == "__main__":
    main()