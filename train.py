import argparse
import datetime
import torch
import torch.nn as nn
import torchsummary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from model import MLP
from dataloader import load_twitter_dataset
import pandas as pd

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, args):
    print('Training...')
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs + 1):
        print('epoch', epoch)
        loss_train = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(device=device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step()

        losses_train.append(loss_train / len(train_loader))

        print('{} Epoch {}, Training loss {:.4f}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader)
        ))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses_train) + 1), losses_train, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # save the loss plot
    plt.savefig(args.save_plot)
    plt.close()


def main():
    # argument parser so train.py can be called using command line
    parser = argparse.ArgumentParser(description='MLP Autoencoder Training')
    parser.add_argument('-z', '--bottleneck', type=int, default=8, help='Bottleneck size')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('-s', '--save-model', type=str, default='MLP.pth', help='Path to save the model')
    parser.add_argument('-p', '--save-plot', type=str, default='loss.png', help='Path to save the loss plot')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 200
    shuffle = True
    num_workers = 4

    train_data = load_twitter_dataset("./Data/twitter_sentiment/twitter.5000.train.json")
    test_data = load_twitter_dataset("./Data/twitter_sentiment/twitter.1000.test.json")
    valid_data = load_twitter_dataset("./Data/twitter_sentiment/twitter.500.valid.json")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    model_layers = [768, 768 // 2, 100, 768 // 2, 768, 2]
    model = MLP(layers=model_layers)
    model.to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    torchsummary.summary(model, model.input_shape)
    losses_train = []
    n_epochs = 100
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch: {epoch}")
        loss_train = 0.0
        for batch_data, batch_targets in train_dataloader:
            batch_data = batch_data.to(device)         # Move data to device
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            #print("Batch Targets: ", batch_targets)
            #print("Outputs: ", outputs)
            loss = loss_fn(batch_targets, outputs)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        scheduler.step()
        losses_train.append(loss_train / len(train_dataloader))

        print(f"{datetime.datetime.now()} Epoch: {epoch}, Loss: {loss_train / len(train_dataloader)}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses_train) + 1), losses_train, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig("./Data/twitter_sentiment/loss.png")
    plt.close()


    # use CUDA if available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # define data transformations and load MNIST dataset
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # # initialize model, loss function, optimizer, and learning rate scheduler
    # model = autoencoderMLP4Layer(N_bottleneck=args.bottleneck)
    # model.to(device)
    # loss_fn = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

    # # print summary
    # torchsummary.summary(model, (1, 28 * 28))

    # # call train function to train model with previously established parameters
    # train(args.epochs, optimizer, model, loss_fn, train_loader, scheduler, device, args)

    # # save the trained model
    # torch.save(model.state_dict(), args.save_model)


if __name__ == '__main__':
    main()
