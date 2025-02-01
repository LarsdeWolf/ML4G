import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import RGCNDataset
from train import train
from model import *

# HyperParameters
EPOCHS = 20
LR = 1e-4
P_DROPOUT = 0.3
WEIGHT_DECAY = 1e-5
W_SHARE = True


def run(version, num_runs=3):
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': []
    }

    path = 'wikihop/'
    train_set, eval_set = (
        RGCNDataset(dataset=f'{path}train.json',
                    entities=f'{path}{version}/train_ents.csv',
                    nquads=f'{path}{version}/train_nquads.csv',
                    embeddings=f'{path}{version}/train_embeds.csv',
                    qembeddings=f'{path}{version}/train_qembeds.csv'),
        RGCNDataset(dataset=f'{path}dev.json',
                    entities=f'{path}{version}/dev_ents.csv',
                    nquads=f'{path}{version}/dev_nquads.csv',
                    embeddings=f'{path}{version}/dev_embeds.csv',
                    qembeddings=f'{path}{version}/dev_qembeds.csv')
    )

    for run in range(1, num_runs + 1):
        SEED = np.random.randint(0, 10000)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model = RGCN(list(train_set.relations), W_SHARE, dropout=P_DROPOUT, n_layers=3)
        optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        loss = nn.CrossEntropyLoss()
        print(f"Run {run} (seed={SEED})")
        epoch_train_loss, epoch_dev_loss, epoch_train_acc, epoch_dev_acc = train(model, [train_set, eval_set], optim, loss, EPOCHS, SEED)
        metrics['train_loss'].append(epoch_train_loss)
        metrics['train_acc'].append(epoch_train_acc)
        metrics['dev_loss'].append(epoch_dev_loss)
        metrics['dev_acc'].append(epoch_dev_acc)

    return metrics


def plot_metrics(metrics, path, version):
    epochs = range(1, 21)
    for metric in metrics.keys():
        for i, values in enumerate(metrics[metric]):
            plt.plot(epochs, metrics, label=f'Run {i + 1}', alpha=0.6)
        metrics_arr = np.array(metrics[metric])
        avg = metrics_arr.mean(axis=0)
        std = metrics_arr.std(axis=0)
        plt.plot(epochs, avg, label='Average', color='black', linewidth=2)
        plt.fill_between(epochs, avg - std, avg + std, color='black', alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel(metric.split('_')[1])
        plt.title(f'{metric.split("_")} per Epoch')
        plt.legend()
        plt.savefig(f'{path}/{metric}_{version}.png')
        plt.close()
        plt.show()


def main():
    os.makedirs('results', exist_ok=True)
    for version in ['normal', 'extended']:
        print("==========================================")
        print(f"Running experiments for dataset version: {version}")
        print("==========================================")
        version_dir = os.path.join('results', version)
        os.makedirs(version_dir, exist_ok=True)
        plot_dir = os.path.join('results', version, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        metrics = run(version, num_runs=3)
        plot_metrics(metrics, plot_dir, version)
    print("All experiments completed.")


if __name__ == '__main__':
    main()
