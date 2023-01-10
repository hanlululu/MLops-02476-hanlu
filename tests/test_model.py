import torch
from src.data.make_dataset import CorruptMnist
from src.models.model import MyAwesomeModel
import pytest

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    batch_size = 128
    # load the training data 
    train_set = CorruptMnist(train=True,in_folder = "data/raw" , out_folder = "data/processed")
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    with pytest.raises((ValueError, RuntimeError)):
        model(torch.randn(1,2,3))

    for images, labels in trainloader:
        # Forward pass 
        outputs = model(images)
        assert images.shape[0] == outputs.shape[0]
        
if __name__ == "__main__":
    test_error_on_wrong_shape()


    
    
    
    