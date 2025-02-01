import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryNodeEncoder(nn.Module):
    """
    Encoder module for encoding the embedded queries and Node (embeddings), consists of:
        1.  Bidirectional LSTM encoder to produce query representation q
            (2 layers of 256 and 128 hidden units respectively)
        2.  1-layered MLP to project nodes to 256-dimensional vectors
        3.  2-layered MLP to project concatenated nodes with q to query-aware mention nodes
            (2 layers of 1024 and 512 hidden units respectively)
    """
    def __init__(self,
                 query_dim=3072,
                 q_lstm1_dim=256,
                 q_lstm2_dim=128,
                 node_dim=3072,
                 node_fc1_dim=256,
                 qnode_fc1_dim=1024,
                 qnode_fc2_dim=512,
                 dropout=0.2):
        super(QueryNodeEncoder, self).__init__()

        self.query_encoder = nn.ModuleList([
            nn.LSTM(input_size=query_dim, hidden_size=q_lstm1_dim, bidirectional=True),
            nn.LSTM(input_size=q_lstm1_dim * 2, hidden_size=q_lstm2_dim, bidirectional=True)
        ])
        self.node_encoder = nn.Sequential(
            nn.Linear(in_features=node_dim, out_features=node_fc1_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.querynode_encoder = nn.Sequential(
            nn.Linear(in_features=node_fc1_dim * 2, out_features=qnode_fc1_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=qnode_fc1_dim, out_features=qnode_fc2_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, q, nodes):
        """
        Forward pass for the QueryNodeEncoder
        Parameters
        ----------
        q: Embedded query (torch.Tensor[float])
        nodes: Embedded nodes (torch.Tensor[float])

        Returns
        -------
        q: Query representation (torch.Tensor[float])
        nodes: Query-aware mention nodes (torch.Tensor[float])
        """
        # 2-layered biLSTM
        for layer in self.query_encoder:
            q, _ = layer(q)
        # 1-layer MLP
        nodes = self.node_encoder(nodes)
        # Concatenate nodes with q
        nodes = torch.cat((q.repeat(nodes.size(0), 1), nodes), dim=-1)
        # 2-layer MLP
        nodes = self.querynode_encoder(nodes)
        return q, nodes


class RGCNLayer(nn.Module):
    """
    Gated RGCN Layer that performs a one-hop message passing operation with respect to each relation.
    """
    def __init__(self,
                 relations,
                 hidden_in=512,
                 hidden_out=512,
                 dropout=0.2):
        super(RGCNLayer, self).__init__()

        # Update weights
        self.W0 = nn.Parameter(torch.empty((hidden_in, hidden_out)))
        self.relation_weights = nn.ParameterDict({
            relation: nn.Parameter(torch.empty((hidden_in, hidden_out))) for relation in relations
        })

        # Gate weights
        self.gate_weight = nn.Parameter(torch.empty((2 * hidden_in, hidden_out)))

        self.dropout = nn.Dropout(dropout)
        self.init_params()

    def forward(self, H, adj_dict):
        """
        Calculates the vectorized forward pass
        Parameters
        ----------
        H: Hidden representations at layer l (N, hidden_in)
        adj_dict: Dict of #relations adjacency matrices, all shape (N, N)

        Returns
        -------
        Hidden representations at layer l + 1
        """
        # Normalization: 1/#node_neighbors. Since fully connected, equal to 1/(N_nodes -1)
        adj_norm = {rel: 1/(H.shape[0] - 1) * A for rel, A in adj_dict.items()}

        # Messages: sum([A_norm @ H @ W_rel for rel in relations])
        messages = torch.stack(
            [torch.spmm(A, H) @ self.relation_weights[rel] for rel, A in adj_norm.items()],
            dim=0
        ).sum(dim=0)

        # Update
        update = H @ self.W0 + messages

        # Gate
        gate = F.sigmoid(
            torch.cat((update, H), dim=1) @ self.gate_weight
        )

        # Hidden units at L+1
        hidden = F.tanh(update) * gate + H * (1 - gate)
        return self.dropout(hidden)

    def init_params(self):
        for param in self.parameters():
            nn.init.kaiming_uniform_(param)


class RGCN(nn.Module):
    """
    RGCN Module
    """
    def __init__(self,
                 relations,
                 layer_weightshare=True,
                 n_layers=3,
                 hidden_in=512,
                 hidden_out=512,
                 output_h1=256,
                 output_h2=128,
                 output_h3=1,
                 query_dim=3072,
                 q_lstm1_dim=256,
                 q_lstm2_dim=128,
                 node_dim=3072,
                 node_fc1_dim=256,
                 qnode_fc1_dim=1024,
                 qnode_fc2_dim=512,
                 dropout=0.2
                 ):
        super(RGCN, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layer_weightshare = layer_weightshare
        self.n_layers = n_layers
        self.p_dropout = dropout

        self.encoder = QueryNodeEncoder(query_dim, q_lstm1_dim, q_lstm2_dim, node_dim,
                                        node_fc1_dim, qnode_fc1_dim, qnode_fc2_dim, self.p_dropout).to(self.device)

        self.layers = nn.ModuleList(
            # Repetition of same RGCNLayer
            [RGCNLayer(relations, hidden_in, hidden_out, self.p_dropout)] * self.n_layers
        ).to(self.device) if self.layer_weightshare else nn.ModuleList(
            # n_layer RGCN layers
            [RGCNLayer(relations, hidden_in, hidden_out, self.p_dropout) for _ in range(self.n_layers)]
        ).to(self.device)

        self.output = nn.Sequential(
            nn.Linear(hidden_out + q_lstm1_dim, output_h1),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(output_h1, output_h2),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(output_h2, output_h3)
        ).to(self.device)


    def forward(self, nodes, q, adj_dict, cand_nodes):
        """
        Forward pass for the RGCN model
        Parameters
        ----------
        nodes: Embedded nodes (torch.Tensor[float])
        q: Embedded query (torch.Tensor[float])
        adj_dict: Dict of #relations sparse adjacency matrices (torch.Tensor[sparse])
        cand_nodes: Dict of candidate nodes for each query (Dict[List[int]])

        Returns
        -------
        out: Output logits (torch.Tensor[float])
        """
        q, nodes = self.encoder(q, nodes)

        for layer in self.layers:
            nodes = layer(nodes, adj_dict)
        nodes = torch.cat((q.repeat(nodes.size(0), 1), nodes), dim=-1)
        out = self.output(nodes)
        out = torch.cat([
            out[candidates].max(dim=0)[0] if out[candidates].shape[0] != 0 else
            torch.zeros(1, requires_grad=True).to(self.device) for candidates in list(cand_nodes.values())
        ])
        return out


if __name__ == '__main__':
    model = RGCN(['DOC', 'COREF', 'MATCH', 'COMPLEMENT'])
    model = model.to(model.device)
