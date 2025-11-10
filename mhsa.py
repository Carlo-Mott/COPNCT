import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,hidden_dim, embed_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.attention = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)
        # self.norm = nn.LayerNorm(embed_dim)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x, prompt=None):

        B,T,C = x.size()
        qkv = self.qkv(x)
        qkv = qkv.view(B,T,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]

        if prompt is not None:
            prefix_k,prefix_v = prompt
            prefix_k = prefix_k.expand(B,-1,-1,-1)
            prefix_v = prefix_v.expand(B,-1,-1,-1)
            k=torch.cat([prefix_k,k],dim=2)
            v=torch.cat([prefix_v,v],dim=2)

        attn_scores = torch.matmul(q,k.transpose(-2,-1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights,v)

        out = attn_output.transpose(1,2).contiguous().view(B,T,C)
        return self.out_proj(out)