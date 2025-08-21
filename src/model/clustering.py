import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

class DynamicStockClustering(nn.Module):
    """
    This module implements dynamic stock clustering based on PCA.
    It extracts latent market segments using principal component analysis,
    computes stock membership probabilities for each segment,
    and assigns stocks to fine-grained subsegments within each segment.
    """
    def __init__(
        self,
        embedding_dim,
        min_var_ratio,
        temperature,
        n_subclusters,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.min_var_ratio = min_var_ratio
        self.temperature = temperature
        self.n_subclusters = n_subclusters
        
    def extract_latent_segments(self, stock_embs):
        hidden_np = stock_embs.detach().cpu().numpy()

        pca = PCA(n_components=min(hidden_np.shape[0], hidden_np.shape[1]))
        pca.fit(hidden_np)

        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        n_clusters = np.argmax(cumulative_var >= self.min_var_ratio) + 1
        components = pca.components_[:n_clusters] 
        self.original_principal_components = components

        pc_tensor = torch.tensor(components, dtype=torch.float32, device=stock_embs.device)
        return pc_tensor
    
    def compute_membership_probability(self, hidden_rep, market_reps):
        current_n_clusters = market_reps.size(0)

        hidden_normalized = F.normalize(hidden_rep, p=2, dim=-1)
        market_normalized = F.normalize(market_reps, p=2, dim=-1)

        similarities = torch.mm(hidden_normalized, market_normalized.t())

        soft_cluster_weights = F.softmax(similarities / self.temperature, dim=1)
            
        return soft_cluster_weights, current_n_clusters, similarities

    def compute_latent_subsegments(self, hidden_rep, soft_cluster_weights, current_n_clusters):
        N = hidden_rep.size(0)

        latent_subsegment_assignments = torch.zeros(
            N, current_n_clusters, self.n_subclusters, 
            device=hidden_rep.device
        )

        for cluster_idx in range(current_n_clusters):
            cluster_weights = soft_cluster_weights[:, cluster_idx].view(-1, 1)

            if cluster_weights.sum() < 1e-6:
                continue

            original_pc = self.original_principal_components[cluster_idx]
            centroid = torch.tensor(original_pc, dtype=torch.float32, device=hidden_rep.device).unsqueeze(0)

            centroid = centroid.squeeze(0)
            weighted_centroid = centroid.view(1, -1)

            stocks_normalized = F.normalize(hidden_rep, p=2, dim=-1)
            centroid_normalized = F.normalize(weighted_centroid, p=2, dim=-1)
            similarities = torch.mm(stocks_normalized, centroid_normalized.t()).squeeze()

            sorted_sims, sort_indices = torch.sort(similarities, descending=True)

            interval_size = max(1, N // self.n_subclusters)
            
            for i in range(self.n_subclusters):
                start_idx = i * interval_size
                end_idx = min((i + 1) * interval_size if i < self.n_subclusters - 1 else N, N)

                if start_idx >= end_idx:
                    continue

                indices_in_interval = sort_indices[start_idx:end_idx]
                latent_subsegment_assignments[indices_in_interval, cluster_idx, i] = 1.0
        
        return latent_subsegment_assignments

    def forward(self, stock_embs: torch.Tensor):
        market_segments = self.extract_latent_segments(stock_embs)
        soft_cluster_weights, current_n_clusters, similarities = self.compute_membership_probability(stock_embs, market_segments)
        latent_subsegment_assignments = self.compute_latent_subsegments(stock_embs, soft_cluster_weights, current_n_clusters)

        return soft_cluster_weights, latent_subsegment_assignments, current_n_clusters, similarities, market_segments
