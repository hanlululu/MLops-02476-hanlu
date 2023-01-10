# from src.data.make_dataset import CorruptMnist
# import torch
# import os.path
# import pytest 

# file_path = 'data/'
# file_path_out = 'data/processed'
# @pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
# @pytest.mark.parametrize("test_input,expected", [(train, 40000), (test, 5000)])

# def test_data(test_input,expected):
#     train = CorruptMnist(train=True,in_folder = file_path , out_folder = file_path_out)
#     test = CorruptMnist(train=False, in_folder= file_path, out_folder=file_path_out)

#     assert len(train) ==  40000
#     assert len(test) == 5000
#     assert (train.images.shape == torch.Size([40000, 1, 28, 28])) & (test.images.shape == torch.Size([5000, 1, 28, 28]))
#     assert (train.labels.shape == torch.Size([40000])) & (test.labels.shape == torch.Size([5000]))

# if __name__ == "__main__":
#     test_data()

from tests import _PATH_DATA
from src.data.make_dataset import CorruptMnist
import torch
import os.path
import pytest

@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/train_processed.pt'), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/test_processed.pt'), reason="Data files not found")
@pytest.mark.parametrize("file,expected_dim", [(0, torch.Size([1,28,28])), (1, torch.Size([]))])

def test_data(file, expected_dim):
    dataset_train = torch.load(f'{_PATH_DATA}/processed/train_processed.pt')
    dataset_test =torch.load(f'{_PATH_DATA}/processed/test_processed.pt')

    assert len(dataset_train[0]) == 40000, "Dataset for training does not have correct dimension"
    assert len(dataset_test[0]) == 5000, "Dataset for test does not have correct dimension"
    for images in dataset_train[file]:
        assert images.shape == expected_dim, "Shape of of input does not fit"
    assert len(dataset_train[0]) == len(dataset_train[1]), "Dataset have uneven number of images and labels"



if __name__ == "__main__":
    test_data()