import argparse
import datetime
import torch
import torch.nn as nn
import torchsummary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from model_parts.base_models import MLP
from dataloader import load_twitter_dataset
import pandas as pd

class HyperParameters:
    def __init__(self):
        pass

def main():
    # argument parser so train.py can be called using command line
    parser = argparse.ArgumentParser(description='MLP Autoencoder Training')
    parser.add_argument('-f', '--lossfile', type=str, default="loss.png", help="FileName of loss file")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    batch_size = 500
    shuffle = True
    num_workers = 4

    train_data = load_twitter_dataset("./Data/twitter_sentiment/twitter.100000.train.json")
    #test_data = load_twitter_dataset("./Data/twitter_sentiment/twitter.10000.test.json")
    valid_data = load_twitter_dataset("./Data/twitter_sentiment/twitter.1000.valid.json")

    print(f"Train Distribution: {train_data.get_target_distribution()}")
    print(f"Validation Distribution: {valid_data.get_target_distribution()}")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    

    model_layers = [768, 768 * 2, 768, 400, 768, 2]
    model = MLP(layers=model_layers)
    model.to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    torchsummary.summary(model, model.input_shape)
    losses_train = []
    losses_valid = []

    n_epochs = 100
    save_path = "./Data/twitter_sentiment/train_results/modelMLP.pth"
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch: {epoch}")
        loss_train = 0.0
        for batch_data, batch_targets in train_dataloader:
            batch_data = batch_data.to(device)         # Move data to device
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = loss_fn(batch_targets, outputs)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        scheduler.step()
        losses_train.append(loss_train / len(train_dataloader))

        loss_valid = 0.0
        model.eval()
        with torch.no_grad():
            for batch_data, batch_targets in valid_dataloader:
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)
                outputs = model(batch_data)
                loss = loss_fn(batch_targets, outputs)
                loss_valid += loss.item()
            losses_valid.append(loss_valid / len(valid_dataloader))
        model.train()
        print(f"{datetime.datetime.now()} Epoch: {epoch}, Train Loss: {loss_train / len(train_dataloader)}, Valid Loss: {loss_valid / len(valid_dataloader)}")
        if losses_valid[-1] < 0.145:
            break
    
    
    model.save_file(save_path)

    plt.figure()
    plt.plot(range(1, len(losses_train) + 1), losses_train, label='Training Loss')
    plt.plot(range(1, len(losses_valid) + 1), losses_valid, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig("./Data/twitter_sentiment/train_results/loss2.png")
    plt.close()

if __name__ == '__main__':
    main()
