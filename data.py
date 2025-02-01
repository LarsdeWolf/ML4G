import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


class RGCNDataset(Dataset):
    """
    Dataset class for the RGCN model.
    Precomputes all data, allowing for fast indexing.
    """
    def __init__(self, dataset, entities, nquads, embeddings, qembeddings):
        self.nquads = pd.read_csv(nquads)
        self.entities = pd.read_csv(entities)
        self.embeddings = pd.read_csv(embeddings, header=None).to_numpy()
        self.q_embeddings = pd.read_csv(qembeddings, header=None).to_numpy()
        self.data = pd.read_json(dataset).iloc[:self.q_embeddings.shape[0]]
        self.N_nodes = len(self.entities)
        self.N_edges = len(self.nquads)

        self.relations = self.nquads['relation'].unique()
        self.adj_cache, self.cand_cache, self.target_cache = {}, {}, {}

        self.indexes = []
        for idx in range(len(self.data)):
            id, query_str, answer, candidates = self.data.iloc[idx][['id', 'query', 'answer', 'candidates']]
            nquads = self.nquads[self.nquads['datapoint_i'] == idx]
            nodes = np.unique(nquads[['subject', 'object']].to_numpy().reshape(1, -1 * 2))
            node_to_i = {node_id: i for i, node_id in enumerate(nodes)}
            cand_nodes = {cand: [node_to_i[node] for node in nodes if self.entities.iloc[node]['entity'] == cand] for
                          cand in candidates}
            adj_matrices = {}

            # Adjacency matrix for each relation
            for rel in self.relations:
                adj = np.zeros((len(nodes), len(nodes)))
                nquads_rel = nquads[nquads['relation'] == rel]
                source = nquads_rel['subject'].apply(lambda x: node_to_i[x]).to_numpy()
                object = nquads_rel['object'].apply(lambda x: node_to_i[x]).to_numpy()
                np.add.at(adj, (source, object), 1)
                np.add.at(adj, (object, source), 1)
                adj_matrices[rel] = torch.Tensor(adj).to_sparse()
            # Target vector
            target = torch.zeros(len(candidates))
            target[candidates.index(answer)] = 1

            self.adj_cache[idx] = adj_matrices
            self.cand_cache[idx] = cand_nodes
            self.target_cache[idx] = target
            self.indexes.append(idx)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        nquads = self.nquads[self.nquads['datapoint_i'] == idx]
        nodes = np.unique(nquads[['subject', 'object']].to_numpy().reshape(1, -1 * 2))

        node_embeddings = self.embeddings[nodes]
        query_emb = self.q_embeddings[idx]
        adj_matrices = self.adj_cache[idx]
        target = self.target_cache[idx]
        cand_nodes = self.cand_cache[idx]
        i_to_node = {i: node_id for i, node_id in enumerate(nodes)}

        return (torch.Tensor(node_embeddings),
                torch.Tensor(query_emb),
                adj_matrices,
                target,
                cand_nodes,
                i_to_node)


