import torch
import torch.nn as nn

class EmbeddingResNet(nn.Module):
    def __init__(self, emb_dims, n_cont, hidden_dim=128, output_dim=1, dropout=0.2):
        super().__init__()
        
        # 1. Embedding layers for each categorical feature
        # emb_dims is a list of (num_categories, embedding_dim) tuples
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        n_emb = sum([dim for _, dim in emb_dims])
        
        # 2. ResNet Backbone
        self.first_layer = nn.Sequential(
            nn.Linear(n_emb + n_cont, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_cat, x_cont):
        # Embed categorical data
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x = torch.cat(embeddings, 1)
        
        # Concatenate with numerical data
        x = torch.cat([x, x_cont], 1)
        
        # Standard ResNet flow
        x = self.first_layer(x)
        identity = x
        out = self.res_block(x)
        out += identity 
        out = self.relu(out)
        return self.head(out)