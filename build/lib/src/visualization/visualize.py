import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import click
from sklearn import manifold

from src.data.make_dataset import CorruptMnist
from src.models.model import MyAwesomeModel

@click.command()
@click.argument('model_checkpoint')
def tsne_embedding_plot(model_checkpoint) -> None:
    train_set = CorruptMnist(train=True, in_folder="data/raw", out_folder="data/processed")
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))

    print("Extract embeddings")
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            # Extract features from the backbone
            emb = model.layer1(x).reshape(x.shape[0], -1)
            embeddings.append(emb)
            labels.append(y)

    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).numpy()

    print("Running tsne")
    tsne = manifold.TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    for i in np.unique(labels):
        plt.scatter(embeddings_2d[labels == i, 0], embeddings_2d[labels == i, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/2d_tsne_embedding.png")


if __name__ == "__main__":
    tsne_embedding_plot()