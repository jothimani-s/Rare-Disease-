import torch
import torch.nn as nn

class AdaptiveBiomedicalAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, weights = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        f = self.ffn(x)
        out = self.norm2(x + f)
        return out, weights
