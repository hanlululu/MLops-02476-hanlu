import argparse
import sys

import torch
import click

from torch import nn, optim
from src.data.make_dataset import CorruptMnist
from tqdm import tqdm
from src.models.model import MyAwesomeModel
import matplotlib.pyplot as plt



@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model.train()
    
    batch_size = 128
    # load the training data 
    train_set = CorruptMnist(train=True,in_folder = "data/raw" , out_folder = "data/processed")
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss() # define type of loss 
    optimizer = optim.Adam(model.parameters(), lr=lr) # define optimizer

    epochs = 5

    train_losses = []
    train_accuracy = []

    for e in range(epochs):
        accuracy = 0 
        with tqdm(trainloader, unit="batch") as tepoch:
            for images, labels in trainloader:
                tepoch.set_description(f"Epoch {e}")

                # Forward pass 
                outputs = model(images)
                ps = torch.exp(outputs)
                loss = criterion(ps, labels) # train loss 
                train_losses.append(loss.item())

                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))/len(trainloader)

                train_accuracy.append(accuracy)

                # Backpropogate 
                loss.backward()

                # Update parameters 
                optimizer.step()

                # Clear gradients 
                optimizer.zero_grad()

                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy.item()*100)

    torch.save(model.state_dict(), 'models/trained_model.pt')

    figure, axis = plt.subplots(2)
  
    # For loss
    axis[0].plot(train_losses,label="loss")
    axis[0].set_title("Training loss")
    axis[0].set_xlabel("iterations")
    axis[0].set_ylabel("loss")
    
    # For accuracy
    axis[1].plot(train_accuracy,label="accuracy")
    axis[1].set_title("Training accuracy")
    axis[0].set_xlabel("iterations")
    axis[0].set_ylabel("loss")

    plt.savefig('reports/figures/training_progress.png')

    return model


if __name__ == "__main__":
    train()


    
    
    
    