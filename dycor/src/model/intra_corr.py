import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelSubclusterAttention(nn.Module):
    """
    Parallel attention mechanism for processing multiple subclusters simultaneously.
    This module applies attention between stocks within the same subcluster,
    processing all subclusters in parallel for efficiency.
    """
    def __init__(self, hidden_size, n_subclusters=6, hidden_dropout_prob=0.1, 
                 attn_dropout_prob=0.1):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_subclusters = n_subclusters

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size, eps=1e-12)
            for _ in range(n_subclusters)
        ])
        
    def reshape_for_attention(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, 1, self.hidden_size).transpose(1, 2)
    
    def forward(self, hidden_rep, combined_attention_mask):
        N = hidden_rep.size(0)

        residual = hidden_rep

        q = self.q_proj(hidden_rep)
        k = self.k_proj(hidden_rep)
        v = self.v_proj(hidden_rep)

        q = q.unsqueeze(1)

        k_expanded = k.unsqueeze(0).expand(N, -1, -1)
        v_expanded = v.unsqueeze(0).expand(N, -1, -1)

        q = self.reshape_for_attention(q)
        k = self.reshape_for_attention(k_expanded)
        v = self.reshape_for_attention(v_expanded)

        q = q / (self.hidden_size ** 0.5)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        outputs = []

        for s in range(self.n_subclusters):
            attn_mask = combined_attention_mask[s]

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

            masked_attention_scores = attention_scores + attn_mask

            attention_weights = F.softmax(masked_attention_scores, dim=-1)
            attention_weights = self.attn_dropout(attention_weights)

            context = torch.matmul(attention_weights, v)

            context = context.transpose(1, 2).contiguous()
            context = context.view(N, 1, self.hidden_size)
            context = context.squeeze(1)

            output = self.o_proj(context)
            output = self.out_dropout(output)

            output = self.layer_norms[s](output + residual)
            
            outputs.append(output)
        
        return outputs

class IntraStockCorrelation(nn.Module):
    """
    Models correlations between stocks within the same subcluster.
    Creates attention masks based on cluster and subcluster memberships,
    and applies self-attention to capture intra-cluster stock relationships.
    """
    def __init__(
        self,
        hidden_dim,
        n_subclusters,
        dropout,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_subclusters = n_subclusters

        self.attention = ParallelSubclusterAttention(
            hidden_size=hidden_dim,
            n_subclusters=n_subclusters,
            hidden_dropout_prob=dropout,
            attn_dropout_prob=dropout,
        )

    def create_soft_attention_masks(self, soft_cluster_weights, latent_subsegment_assignments, N, current_n_clusters, device):
        attention_masks = []

        for k in range(self.n_subclusters):
            attn_mask = torch.zeros((N, N), device=device)

            for c in range(current_n_clusters):
                cluster_weight_i = soft_cluster_weights[:, c].view(N, 1)
                cluster_weight_j = soft_cluster_weights[:, c].view(1, N)

                subcluster_weight_i = latent_subsegment_assignments[:, c, k].view(N, 1)
                subcluster_weight_j = latent_subsegment_assignments[:, c, k].view(1, N)

                cluster_similarity = cluster_weight_i * cluster_weight_j
                subcluster_similarity = subcluster_weight_i * subcluster_weight_j

                attn_mask += cluster_similarity * subcluster_similarity

            attn_mask = attn_mask / (attn_mask.max() + 1e-8)

            attn_mask = torch.log(attn_mask + 1e-10)

            attention_masks.append(attn_mask.view(N, 1, 1, N))
            
        return attention_masks

    def forward(
        self,
        stock_embs,
        soft_cluster_weights,
        latent_subsegment_assignments,
        current_n_clusters
    ):
        N = stock_embs.size(0)

        attention_masks = self.create_soft_attention_masks(
            soft_cluster_weights, latent_subsegment_assignments, N, current_n_clusters, stock_embs.device
        )

        combined_masks = torch.stack(attention_masks)

        interval_outputs = self.attention(stock_embs, combined_masks)

        return interval_outputs
