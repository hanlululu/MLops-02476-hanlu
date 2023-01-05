import argparse
import sys
import os

import torch
import click
import numpy as np

from torch import nn
from src.models.model import MyAwesomeModel



@click.command()
@click.argument("trained_model")
@click.argument("test_data")
def evaluate(trained_model,test_data):
    print("Evaluating until hitting the ceiling")
    print(trained_model)

#   # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    checkpoint = torch.load(trained_model)
    model.load_state_dict(checkpoint)
    
    model.eval()

    batch_size = 128
    # load the training data 
    test_set = torch.tensor(np.load(test_data))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss() # define type of loss 

    test_losses = []
    test_accuracy = []
    accuracy = 0 
    with torch.no_grad():
        for images, labels in testloader:
            # Forward pass 
            outputs = model(images)
            ps = torch.exp(outputs)
            loss = criterion(ps, labels) # test loss 
            test_losses.append(loss.item())

            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))/len(testloader)

            test_accuracy.append(accuracy)
    
    print(f'Accuracy: {accuracy.item()*100}%')


if __name__ == "__main__":
    evaluate()