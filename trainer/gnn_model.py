import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    class GATConv: pass
    class Data: pass
    class Batch: pass

class DocumentGNN(nn.Module):
    def __init__(self, node_in_dim=33, edge_in_dim=51, hidden_dim=128, num_classes=11, heads=4):
        """
        GNN model for Document Parsing.
        Predicts node classes (for F1 element classification) 
        and edge scores (for reading order KTDS).
        """
        super(DocumentGNN, self).__init__()
        if not PYG_AVAILABLE:
            raise RuntimeError("PyTorch Geometric is not installed")
            
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim // heads)
        
        # GAT Layers
        self.conv1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=hidden_dim // heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=hidden_dim // heads)
        
        # Node classification head
        self.node_cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Edge scoring head (reading order link prediction)
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        x: [num_nodes, node_in_dim]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_in_dim]
        """
        if edge_attr is None:
            # Fallback if no edge features provided natively
            edge_attr = torch.zeros(edge_index.size(1), self.edge_proj.in_features, device=x.device)
            
        # 1. Project inputs
        h = F.relu(self.node_proj(x))
        e = F.relu(self.edge_proj(edge_attr))
        
        # 2. Message Passing
        # GAT uses edge_attr correctly in PyG
        h = F.relu(self.conv1(h, edge_index, edge_attr=e))
        h = F.relu(self.conv2(h, edge_index, edge_attr=e))
        
        # 3. Node Predictions
        node_logits = self.node_cls(h)
        
        # 4. Edge Predictions (Reading Order)
        # Concatenate src node, dst node, and original edge features
        src, dst = edge_index
        edge_feat = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        edge_scores = self.edge_scorer(edge_feat).squeeze(-1)
        
        return node_logits, edge_scores
        
def build_graph_from_blocks(blocks, block_schema, pair_schema, device='cpu'):
    """
    Utility to convert a list of blocks and their pairwise features into PyG Data.
    We assume 'block_schema' describes feature keys in block['features'].
    """
    if not PYG_AVAILABLE or not blocks:
        return None
        
    num_nodes = len(blocks)
    x = []
    
    # Simple block node features extracting
    for b in blocks:
        feats = []
        # Fallback dummy features if actual pipeline drops these dynamically
        b_feats = b.get("features", {})
        for name in block_schema:
            feats.append(float(b_feats.get(name, 0.0)))
        x.append(feats)
        
    x = torch.tensor(x, dtype=torch.float, device=device)
    
    # We fully connect nodes with basic top-down heuristics or k-NN initially
    # For training we'd use the provided ground truth adjacencies via pairs
    edge_index = []
    edge_attr = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j: continue
            edge_index.append([i, j])
            # In actual `eval.py` we compute advanced pairwise features
            # edge_attr.append(pair_features) - omitted for simplicity in struct creation
            edge_attr.append([0.0] * len(pair_schema))
            
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, len(pair_schema)), dtype=torch.float, device=device)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=device)
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
