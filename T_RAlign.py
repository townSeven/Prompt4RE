import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class TRAlign(nn.Module):
    def __init__(self, region_dim=144, llm_dim=3584, dropout=0.1, num_heads=4):
        super(TRAlign, self).__init__()
        assert region_dim % num_heads == 0, "region_dim must be divisible by num_heads"

        self.region_dim = region_dim
        self.llm_dim = llm_dim
        self.num_heads = num_heads
        self.head_dim = region_dim // num_heads

        # 多头投影
        self.wq = nn.Linear(region_dim, region_dim)
        self.wk = nn.Linear(llm_dim, region_dim)
        # 每个 head 的 V 都有独立线性映射
        self.wv_heads = nn.ModuleList([nn.Linear(llm_dim, llm_dim // num_heads) for _ in range(num_heads)])

        # 残差映射 target_emb -> llm_dim
        if region_dim != llm_dim:
            self.residual_proj = nn.Linear(region_dim, llm_dim)
        else:
            self.residual_proj = nn.Identity()

        self.out_proj = nn.Linear(llm_dim, llm_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(llm_dim)

    def forward(self, target_emb, source_emb, value_emb):
        """
        target_emb: [N, region_dim]
        source_emb: [M, llm_dim]
        value_emb:  [M, llm_dim]
        输出: [N, llm_dim]
        """
        N, M = target_emb.size(0), source_emb.size(0)

        # 投影
        Q = self.wq(target_emb)       # [N, region_dim]
        K = self.wk(source_emb)       # [M, region_dim]

        # 多头切分 Q/K
        Q_heads = Q.view(N, self.num_heads, self.head_dim).transpose(0,1)  # [num_heads, N, head_dim]
        K_heads = K.view(M, self.num_heads, self.head_dim).transpose(0,1)  # [num_heads, M, head_dim]

        outputs = []
        for i in range(self.num_heads):
            V_i = self.wv_heads[i](value_emb)  # [M, llm_dim//num_heads]
            attn_scores = torch.matmul(Q_heads[i], K_heads[i].T) / sqrt(self.head_dim)  # [N, M]
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out_i = torch.matmul(attn_weights, V_i)  # [N, llm_dim//num_heads]
            outputs.append(out_i)

        # 合并多头
        output = torch.cat(outputs, dim=-1)  # [N, llm_dim]

        # 残差 + LayerNorm
        output = self.layernorm(output + self.residual_proj(target_emb))

        # 输出线性映射
        output = self.out_proj(output)  # [N, llm_dim]

        return output
