import torch
import sys
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
from data import RGCNDataset
from model import RGCN


def train(model, data, optim, criterion, epochs, SEED, log=False):
    train_data, dev_data = data
    epoch_train_loss, epoch_dev_loss, epoch_train_acc, epoch_dev_acc = [], [], [], []
    if log:
        print("-----------------------------------------------------------------------------------------------")
        print(f"Training with {len(train_data)} points and evaluation with {len(dev_data)} points")
        print(f"Train nodes: {train_data.N_nodes}\tTrain edges: {train_data.N_edges}\tDev nodes: {dev_data.N_nodes}\tDev edges: {dev_data.N_edges}")
        print(f"relations: {train_data.relations}")
        print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} Trainable parameters")
        print(f"SEED: {SEED}\tLR: {optim.defaults['lr']}\tP_DROPOUT: {model.p_dropout}\tN_EPOCHS: {epochs}")
        print("-----------------------------------------------------------------------------------------------")

    bar = tqdm(total=len(train_data), desc="Training Progress", position=0, leave=True, file=sys.stdout)
    for EPOCH in range(epochs):
        indexes = train_data.indexes
        np.random.shuffle(indexes)
        train_loss = 0
        train_acc = 0

        model.train()
        for i, idx in enumerate(indexes):
            node_emb, query_emb, adj_matrices, target, cand_nodes, i_to_node = train_data[idx]
            node_emb, query_emb, adj_matrices, target = (
                node_emb.to(model.device),
                query_emb.to(model.device),
                {rel: A.to(model.device) for rel, A in adj_matrices.items()},
                target.to(model.device)
            )

            optim.zero_grad()
            logits = model(node_emb, query_emb.reshape(1, -1), adj_matrices, cand_nodes)
            loss = criterion(logits, target)
            loss.backward()
            optim.step()

            train_loss += loss.item()
            train_acc += (target.argmax() == logits.argmax()).item()

            bar.update(1)
            bar.set_postfix(epoch=EPOCH + 1, loss=loss.item())

        dev_loss = 0
        dev_acc = 0

        model.eval()
        indexes = dev_data.indexes
        for idx in indexes:
            with torch.no_grad():
                node_emb, query_emb, adj_matrices, target, cand_nodes, i_to_node = dev_data[idx]
                node_emb, query_emb, adj_matrices, target = (
                    node_emb.to(model.device),
                    query_emb.to(model.device),
                    {rel: A.to(model.device) for rel, A in adj_matrices.items()},
                    target.to(model.device)
                )
                logits = model(node_emb, query_emb.reshape(1, -1), adj_matrices, cand_nodes)
                loss = criterion(logits, target)

                dev_loss += loss.item()
                dev_acc += (target.argmax() == logits.argmax()).item()
        if log:
            print(f"\nAVG TRAIN Loss: {train_loss / len(train_data)}\nAVG TRAIN Accuracy: {train_acc/len(train_data)}"
                  f"\nAVG DEV Loss: {dev_loss / len(dev_data)}\nAVG DEV Accuracy: {dev_acc/len(dev_data)}")
            print("-----------------------------------------------------------------------------------------------")
        epoch_train_loss.append(train_loss / len(train_data))
        epoch_dev_loss.append(dev_loss / len(dev_data))
        epoch_train_acc.append(train_acc / len(train_data))
        epoch_dev_acc.append(dev_acc / len(dev_data))
        bar.reset()

    bar.close()
    return epoch_train_loss, epoch_dev_loss, epoch_train_acc, epoch_dev_acc


if __name__ == '__main__':
    SEED = np.random.randint(0, 10000)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    RELATIONS = ['DOC', 'COREF', 'MATCH', 'COMPLEMENT', 'SEMANTIC']
    W_SHARE = True
    LR = 1e-4
    EPOCHS = 20
    P_DROPOUT = 0.3

    PATH = 'wikihop/'
    version = 'normal/' # 'extended/'
    data = []
    for SPLIT in ['train', 'dev']:
        ENT_PATH = f'{PATH}{version}{SPLIT}_ents.csv'
        REL_PATH = f'{PATH}{version}{SPLIT}_nquads.csv'
        EMB_PATH = f'{PATH}{version}{SPLIT}_embeds.csv'
        QEMB_PATH = f'{PATH}{version}{SPLIT}_qembeds.csv'
        DATA = f'{PATH}{SPLIT}.json'
        data.append(RGCNDataset(DATA, ENT_PATH, REL_PATH, EMB_PATH, QEMB_PATH))
    model = RGCN(RELATIONS, W_SHARE, dropout=P_DROPOUT, n_layers=3)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    loss = nn.CrossEntropyLoss()
    epoch_train_loss, epoch_dev_loss, epoch_train_acc, epoch_dev_acc = train(model, data, optim, loss, EPOCHS, SEED)