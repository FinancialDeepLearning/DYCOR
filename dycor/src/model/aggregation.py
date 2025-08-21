import torch
import torch.nn as nn

class InterClusterAggregator(nn.Module):
    """
    Aggregates stock representations from multiple clusters and subclusters.
    This module combines different views of each stock (from different market segments)
    weighted by their cluster and subcluster membership probabilities.
    """
    def __init__(
        self,
        hidden_dim,
        n_subclusters,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_subclusters = n_subclusters

    def forward(
        self,
        stock_embs,
        soft_cluster_weights,
        latent_subsegment_assignments,
        interval_outputs,
        current_n_clusters
    ):
        N = stock_embs.size(0)

        combined_weights = torch.zeros(N, current_n_clusters, self.n_subclusters, device=stock_embs.device)

        for k in range(current_n_clusters):
            for l in range(self.n_subclusters):
                combined_weights[:, k, l] = soft_cluster_weights[:, k] * latent_subsegment_assignments[:, k, l]

        combined_weights = combined_weights.sum(dim=1)

        final_stock_reps = torch.zeros_like(stock_embs)

        for l in range(self.n_subclusters):
            final_stock_reps += combined_weights[:, l].unsqueeze(-1) * interval_outputs[l]

        return final_stock_reps