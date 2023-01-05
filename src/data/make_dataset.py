import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch 
import numpy as np
from torch.utils.data import Dataset 
import torchvision.transforms as transforms


# Define a class for our data
class CorruptMnist(Dataset):
    def __init__(self, train: bool, in_folder: str = "", out_folder: str = "") -> None:
        super().__init__()
        
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.train = train

        # Define transformer to normalize the data

        if self.out_folder: # try loading the preprocessed data 
            try: 
                self.load_preprocessed()
                print("Loaded the pre-processed files")
                return
            except ValueError: # Data not created yet and we will create it 
                pass

        # Loading the data 
        if self.train:
            # load all training datasets
            images = []
            labels = []

            # Loop through each of the files to extract the data
            for i in range(0,5):
                with np.load(self.in_folder+'/train_' +str(i)+'.npz') as f:
                    images_tensor = torch.tensor(f['images'])
                    mean = torch.mean(images_tensor,[1,2])
                    std = torch.std(images_tensor,[1,2])
                    transform = transforms.Compose([transforms.Normalize(mean,std)])
                    images.append(transform(images_tensor))
                    labels.append(f['labels'])

            images = torch.tensor(np.concatenate([c for c in images])).reshape(-1, 1, 28, 28)
            labels = torch.tensor(np.concatenate([c for c in labels]))
        else:
        # Load the validation data 
            with np.load(self.in_folder+'/test.npz') as f:
                images_test, labels_test = f['images'], f['labels']
                images_tensor = torch.tensor(images_test)
                mean = torch.mean(images_tensor,[1,2])
                std = torch.std(images_tensor,[1,2])
                transform = transforms.Compose([transforms.Normalize(mean,std)])
                images = transform(images_tensor).reshape(-1, 1, 28, 28)
                labels = torch.from_numpy(labels_test)

        self.images = images
        self.labels = labels
        self.transform = transform 

        if self.out_folder:
            self.save_preprocessed()
    
    def save_preprocessed(self) -> None:
        split = "train" if self.train else "test"
        torch.save([self.images, self.labels], f"{self.out_folder}/{split}_processed.pt")

    def load_preprocessed(self) -> None:
        split = "train" if self.train else "test"
        try:
            self.images, self.labels = torch.load(f"{self.out_folder}/{split}_processed.pt")
        except:
            raise ValueError("No preprocessed files found")


    def __len__(self):
        return self.labels.numel()
        
    def __getitem__(self, idx):
        x = self.images[idx].float()
        y = self.labels[idx]

        return x,y


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train = CorruptMnist(train=True, in_folder=input_filepath, out_folder=output_filepath)
    train.save_preprocessed()

    test = CorruptMnist(train=False, in_folder=input_filepath, out_folder=output_filepath)
    test.save_preprocessed()

    print(train.images.shape)
    print(train.labels.shape)
    print(test.images.shape)
    print(test.labels.shape)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
