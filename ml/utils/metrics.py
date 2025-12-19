"""
Evaluation metrics for GraphVAE.

Implements metrics for:
    1. Reconstruction quality (topology, edge features, transfer function)
    2. Latent space quality (clustering, GED correlation)
    3. Generation quality (novelty, validity, diversity)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import pearsonr, spearmanr
from torch_geometric.data import Data


def chamfer_distance_numpy(
    pred_points: np.ndarray,
    target_points: np.ndarray
) -> float:
    """
    Compute Chamfer distance between two point sets (NumPy version).

    Args:
        pred_points: Predicted points [N1, D]
        target_points: Target points [N2, D]

    Returns:
        distance: Scalar Chamfer distance
    """
    if len(pred_points) == 0 and len(target_points) == 0:
        return 0.0

    if len(pred_points) == 0:
        return float(np.sum(target_points**2))

    if len(target_points) == 0:
        return float(np.sum(pred_points**2))

    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(pred_points, target_points, metric='euclidean')**2

    # Chamfer distance
    min_pred_to_target = dist_matrix.min(axis=1).mean()
    min_target_to_pred = dist_matrix.min(axis=0).mean()

    return float(min_pred_to_target + min_target_to_pred)


class ReconstructionMetrics:
    """
    Metrics for evaluating reconstruction quality.
    """

    @staticmethod
    def topology_accuracy(
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor
    ) -> float:
        """
        Compute topology classification accuracy.

        Args:
            pred_logits: Predicted topology logits [B, 6]
            target_labels: Target filter types (one-hot) [B, 6]

        Returns:
            accuracy: Fraction of correct predictions
        """
        pred_labels = pred_logits.argmax(dim=-1)
        target_labels_idx = target_labels.argmax(dim=-1)

        correct = (pred_labels == target_labels_idx).float().sum()
        total = len(pred_labels)

        return float(correct / total)

    @staticmethod
    def edge_feature_mae(
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute Mean Absolute Error for edge features.

        Args:
            pred_features: Predicted edge features [E, D]
            target_features: Target edge features [E, D]
            mask: Optional mask for valid edges [E]

        Returns:
            mae: Mean absolute error
        """
        if mask is not None:
            pred_features = pred_features[mask]
            target_features = target_features[mask]

        mae = (pred_features - target_features).abs().mean()
        return float(mae)

    @staticmethod
    def pole_zero_chamfer(
        pred_poles: List[np.ndarray],
        pred_zeros: List[np.ndarray],
        target_poles: List[np.ndarray],
        target_zeros: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute Chamfer distance for poles and zeros.

        Args:
            pred_poles: List of predicted pole arrays [num_poles, 2]
            pred_zeros: List of predicted zero arrays
            target_poles: List of target pole arrays
            target_zeros: List of target zero arrays

        Returns:
            Dictionary with pole/zero Chamfer distances
        """
        pole_distances = []
        zero_distances = []

        for i in range(len(pred_poles)):
            # Poles
            if len(pred_poles[i]) > 0 or len(target_poles[i]) > 0:
                cd_pole = chamfer_distance_numpy(pred_poles[i], target_poles[i])
                pole_distances.append(cd_pole)

            # Zeros
            if len(pred_zeros[i]) > 0 or len(target_zeros[i]) > 0:
                cd_zero = chamfer_distance_numpy(pred_zeros[i], target_zeros[i])
                zero_distances.append(cd_zero)

        return {
            'pole_chamfer': float(np.mean(pole_distances)) if pole_distances else 0.0,
            'zero_chamfer': float(np.mean(zero_distances)) if zero_distances else 0.0,
            'avg_chamfer': float(np.mean(pole_distances + zero_distances)) if (pole_distances or zero_distances) else 0.0
        }

    @staticmethod
    def frequency_response_mse(
        pred_response: torch.Tensor,
        target_response: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute MSE for frequency response (magnitude and phase).

        Args:
            pred_response: Predicted response [B, N, 2] (mag, phase)
            target_response: Target response [B, N, 2]

        Returns:
            Dictionary with magnitude and phase MSE
        """
        mag_mse = ((pred_response[:, :, 0] - target_response[:, :, 0])**2).mean()
        phase_mse = ((pred_response[:, :, 1] - target_response[:, :, 1])**2).mean()

        return {
            'magnitude_mse': float(mag_mse),
            'phase_mse': float(phase_mse),
            'total_mse': float(mag_mse + phase_mse)
        }


class LatentSpaceMetrics:
    """
    Metrics for evaluating latent space quality.
    """

    @staticmethod
    def silhouette_score_by_filter_type(
        latent_vectors: np.ndarray,
        filter_types: np.ndarray
    ) -> float:
        """
        Compute Silhouette score for clustering by filter type.

        Args:
            latent_vectors: Latent representations [N, latent_dim]
            filter_types: Filter type labels [N]

        Returns:
            silhouette: Silhouette coefficient [-1, 1]
        """
        if len(np.unique(filter_types)) < 2:
            return 0.0

        try:
            score = silhouette_score(latent_vectors, filter_types)
            return float(score)
        except:
            return 0.0

    @staticmethod
    def ged_correlation(
        latent_vectors: np.ndarray,
        ged_matrix: np.ndarray,
        indices: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute correlation between latent distance and GED.

        Measures whether similar latent vectors correspond to similar circuits.

        Args:
            latent_vectors: Latent representations [N, latent_dim]
            ged_matrix: Graph edit distance matrix [M, M]
            indices: Optional indices into ged_matrix [N]

        Returns:
            Dictionary with Pearson and Spearman correlations
        """
        N = len(latent_vectors)

        if indices is None:
            indices = np.arange(N)

        # Compute pairwise latent distances
        from scipy.spatial.distance import pdist, squareform
        latent_dist = squareform(pdist(latent_vectors, metric='euclidean'))

        # Get corresponding GED distances
        ged_subset = ged_matrix[np.ix_(indices, indices)]

        # Flatten upper triangle (exclude diagonal)
        triu_indices = np.triu_indices(N, k=1)
        latent_dist_flat = latent_dist[triu_indices]
        ged_dist_flat = ged_subset[triu_indices]

        # Compute correlations
        if len(latent_dist_flat) > 1:
            pearson_corr, _ = pearsonr(latent_dist_flat, ged_dist_flat)
            spearman_corr, _ = spearmanr(latent_dist_flat, ged_dist_flat)
        else:
            pearson_corr = 0.0
            spearman_corr = 0.0

        return {
            'pearson_correlation': float(pearson_corr),
            'spearman_correlation': float(spearman_corr)
        }

    @staticmethod
    def cluster_purity(
        latent_vectors: np.ndarray,
        filter_types: np.ndarray,
        n_clusters: int = 6
    ) -> float:
        """
        Compute cluster purity using k-means.

        Args:
            latent_vectors: Latent representations [N, latent_dim]
            filter_types: True filter type labels [N]
            n_clusters: Number of clusters to use

        Returns:
            purity: Fraction of correctly clustered samples
        """
        from sklearn.cluster import KMeans

        if len(latent_vectors) < n_clusters:
            return 0.0

        # Run k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_vectors)

        # Compute purity
        purity = 0
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.sum() == 0:
                continue
            # Most common true label in this cluster
            cluster_types = filter_types[cluster_mask]
            most_common = np.bincount(cluster_types).argmax()
            purity += (cluster_types == most_common).sum()

        purity = purity / len(latent_vectors)
        return float(purity)

    @staticmethod
    def latent_space_coverage(
        latent_vectors: np.ndarray,
        percentile: float = 95
    ) -> Dict[str, float]:
        """
        Measure coverage of latent space.

        Args:
            latent_vectors: Latent representations [N, latent_dim]
            percentile: Percentile for radius calculation

        Returns:
            Dictionary with coverage statistics
        """
        # Center of mass
        centroid = latent_vectors.mean(axis=0)

        # Distances from centroid
        distances = np.linalg.norm(latent_vectors - centroid, axis=1)

        # Statistics
        mean_dist = float(distances.mean())
        std_dist = float(distances.std())
        max_dist = float(distances.max())
        percentile_dist = float(np.percentile(distances, percentile))

        return {
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'max_distance': max_dist,
            f'p{percentile}_distance': percentile_dist
        }


class GenerationMetrics:
    """
    Metrics for evaluating generated circuits.
    """

    @staticmethod
    def novelty_score(
        generated_circuits: List[Data],
        training_circuits: List[Data],
        ged_threshold: float = 0.5
    ) -> float:
        """
        Compute novelty: fraction of generated circuits far from training set.

        Args:
            generated_circuits: List of generated circuit graphs
            training_circuits: List of training circuit graphs
            ged_threshold: GED threshold for considering "novel"

        Returns:
            novelty: Fraction of novel circuits
        """
        # This requires GED computation which is expensive
        # For now, return a placeholder
        # TODO: Implement efficient GED computation or use learned distance
        return 0.0

    @staticmethod
    def validity_score(
        edge_features: torch.Tensor,
        min_value: float = 1e-12,
        max_value: float = 1e12
    ) -> float:
        """
        Compute validity: fraction of circuits with valid component values.

        Args:
            edge_features: Edge features [E, 3] (log-scale impedance)
            min_value: Minimum valid component value
            max_value: Maximum valid component value

        Returns:
            validity: Fraction of valid components
        """
        # Check if all values are in valid range
        # Note: features are log-scaled, so check log-range
        log_min = np.log(min_value)
        log_max = np.log(max_value)

        valid_mask = (edge_features >= log_min) & (edge_features <= log_max)
        validity = valid_mask.all(dim=-1).float().mean()

        return float(validity)

    @staticmethod
    def diversity_score(
        latent_vectors: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute diversity of generated latent vectors.

        Args:
            latent_vectors: Generated latent vectors [N, latent_dim]

        Returns:
            Dictionary with diversity metrics
        """
        from scipy.spatial.distance import pdist

        if len(latent_vectors) < 2:
            return {'mean_pairwise_distance': 0.0, 'std_pairwise_distance': 0.0}

        # Pairwise distances
        pairwise_dists = pdist(latent_vectors, metric='euclidean')

        return {
            'mean_pairwise_distance': float(pairwise_dists.mean()),
            'std_pairwise_distance': float(pairwise_dists.std()),
            'min_pairwise_distance': float(pairwise_dists.min()),
            'max_pairwise_distance': float(pairwise_dists.max())
        }


class MetricsAggregator:
    """
    Aggregate all metrics for comprehensive evaluation.
    """

    def __init__(self):
        self.reconstruction = ReconstructionMetrics()
        self.latent_space = LatentSpaceMetrics()
        self.generation = GenerationMetrics()

    def compute_all_metrics(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        dataloader,
        device: str = 'cpu',
        ged_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics on a dataset.

        Args:
            encoder: Trained encoder model
            decoder: Trained decoder model
            dataloader: DataLoader for evaluation
            device: Device to run on
            ged_matrix: Optional GED matrix for correlation analysis

        Returns:
            Dictionary with all metrics
        """
        encoder.eval()
        decoder.eval()

        # Collect predictions and targets
        all_latent = []
        all_filter_types = []
        all_indices = []

        all_topo_logits = []
        all_topo_targets = []

        all_pred_poles = []
        all_pred_zeros = []
        all_target_poles = []
        all_target_zeros = []

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch['graph'] = batch['graph'].to(device)
                batch['filter_type'] = batch['filter_type'].to(device)
                batch['poles'] = [p.to(device) for p in batch['poles']]
                batch['zeros'] = [z.to(device) for z in batch['zeros']]

                # Encode
                z, mu, logvar = encoder(
                    batch['graph'].x,
                    batch['graph'].edge_index,
                    batch['graph'].edge_attr,
                    batch['graph'].batch,
                    batch['poles'],
                    batch['zeros']
                )

                # Decode
                decoder_output = decoder(z, hard=False)

                # Store results
                all_latent.append(z.cpu().numpy())
                all_filter_types.append(batch['filter_type'].argmax(dim=-1).cpu().numpy())
                if 'idx' in batch:
                    all_indices.append(batch['idx'].cpu().numpy())

                all_topo_logits.append(decoder_output['topo_logits'].cpu())
                all_topo_targets.append(batch['filter_type'].cpu())

                # Poles and zeros
                pred_poles = decoder_output['poles'].cpu().numpy()
                pred_zeros = decoder_output['zeros'].cpu().numpy()

                for i in range(len(batch['poles'])):
                    # Filter out padding
                    pred_poles_i = pred_poles[i]
                    pred_mag = np.sqrt((pred_poles_i**2).sum(axis=-1))
                    pred_poles_i = pred_poles_i[pred_mag > 1e-6]
                    all_pred_poles.append(pred_poles_i)

                    pred_zeros_i = pred_zeros[i]
                    pred_mag = np.sqrt((pred_zeros_i**2).sum(axis=-1))
                    pred_zeros_i = pred_zeros_i[pred_mag > 1e-6]
                    all_pred_zeros.append(pred_zeros_i)

                    all_target_poles.append(batch['poles'][i].cpu().numpy())
                    all_target_zeros.append(batch['zeros'][i].cpu().numpy())

        # Concatenate results
        latent_vectors = np.vstack(all_latent)
        filter_types = np.concatenate(all_filter_types)

        topo_logits = torch.cat(all_topo_logits, dim=0)
        topo_targets = torch.cat(all_topo_targets, dim=0)

        # Compute metrics
        metrics = {}

        # Reconstruction metrics
        metrics['topology_accuracy'] = self.reconstruction.topology_accuracy(
            topo_logits, topo_targets
        )

        pz_metrics = self.reconstruction.pole_zero_chamfer(
            all_pred_poles, all_pred_zeros,
            all_target_poles, all_target_zeros
        )
        metrics.update(pz_metrics)

        # Latent space metrics
        metrics['silhouette_score'] = self.latent_space.silhouette_score_by_filter_type(
            latent_vectors, filter_types
        )

        metrics['cluster_purity'] = self.latent_space.cluster_purity(
            latent_vectors, filter_types
        )

        coverage = self.latent_space.latent_space_coverage(latent_vectors)
        metrics.update(coverage)

        # GED correlation if available
        if ged_matrix is not None and len(all_indices) > 0:
            indices = np.concatenate(all_indices)
            ged_corr = self.latent_space.ged_correlation(
                latent_vectors, ged_matrix, indices
            )
            metrics.update(ged_corr)

        return metrics
