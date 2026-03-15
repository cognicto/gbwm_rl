"""
Sentiment-Aware Value Network (Critic) for GBWM

This module implements sentiment-aware critic networks that estimate
state values considering market sentiment information.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any

from .feature_encoders import create_encoder, PerFeatureAttentionEncoder


class SentimentAwareValueNetwork(nn.Module):
    """
    Sentiment-aware value network for GBWM
    
    Estimates state values V(s) where state includes sentiment features:
    - State: [time, wealth, vix_sentiment, vix_momentum]
    - Output: Scalar value estimation
    
    Architecture:
        State Encoder → Value Layers → Scalar Output
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64,
        encoder_type: str = "feature",
        num_layers: int = 2,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0
    ):
        """
        Initialize sentiment-aware value network
        
        Args:
            state_dim: Input state dimensionality (2 for baseline, 4 for sentiment)
            hidden_dim: Hidden layer dimension
            encoder_type: Type of encoder ('feature', 'simple', 'adaptive', 'attention', 'per_feature_attention')
            num_layers: Number of value layers after encoding
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0.0 to disable)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        
        # State encoder (shared structure with policy network)
        self.state_encoder = create_encoder(
            encoder_type=encoder_type,
            input_dim=state_dim,
            hidden_dim=hidden_dim
        )

        # Get actual encoder output dimension (may differ from hidden_dim)
        encoder_output_dim = self.state_encoder.get_output_dim()

        # Value estimation layers - first layer bridges encoder output to hidden_dim
        value_layers = []

        for i in range(num_layers):
            input_size = encoder_output_dim if i == 0 else hidden_dim
            value_layers.append(nn.Linear(input_size, hidden_dim))
            
            if use_batch_norm:
                value_layers.append(nn.BatchNorm1d(hidden_dim))
            
            if i < num_layers - 1:  # No activation on final layer
                value_layers.append(nn.Tanh())
                
            if dropout_rate > 0 and i < num_layers - 1:
                value_layers.append(nn.Dropout(dropout_rate))
        
        self.value_layers = nn.Sequential(*value_layers)
        
        # Final value output
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for value function learning"""
        # Value layers
        for layer in self.value_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        # Value head - unit gain for final layer
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network
        
        Args:
            state: (batch_size, state_dim) state tensor
            
        Returns:
            values: (batch_size,) estimated state values
        """
        # Encode state features
        encoded_state = self.state_encoder(state)
        
        # Process through value layers
        value_features = self.value_layers(encoded_state)
        
        # Generate value estimate
        values = self.value_head(value_features)
        
        # Squeeze to remove last dimension: (batch_size, 1) → (batch_size,)
        values = values.squeeze(-1)
        
        return values
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to get value estimates
        
        Args:
            state: Input state tensor
            
        Returns:
            Value estimates
        """
        return self.forward(state)


class DualHeadValueNetwork(nn.Module):
    """
    Dual-head value network with separate wealth and goal value estimation
    
    Estimates two components of value:
    1. Wealth value: Expected future wealth accumulation
    2. Goal value: Expected future goal utilities
    
    This decomposition can help with interpretability and training stability.
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64,
        encoder_type: str = "feature"
    ):
        """
        Initialize dual-head value network
        
        Args:
            state_dim: Input state dimensionality
            hidden_dim: Hidden layer dimension
            encoder_type: Type of encoder to use
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        
        # Shared state encoder
        self.state_encoder = create_encoder(
            encoder_type=encoder_type,
            input_dim=state_dim,
            hidden_dim=hidden_dim
        )

        # Get actual encoder output dimension (may differ from hidden_dim)
        encoder_output_dim = self.state_encoder.get_output_dim()

        # Shared processing layers - first layer bridges encoder output to hidden_dim
        self.shared_layers = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Wealth value head
        self.wealth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Goal value head
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Combination weights (learnable)
        self.wealth_weight = nn.Parameter(torch.tensor(0.5))
        self.goal_weight = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in [self.shared_layers, self.wealth_head, self.goal_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual-head value network
        
        Args:
            state: Input state tensor
            
        Returns:
            Combined value estimates
        """
        # Encode state
        encoded_state = self.state_encoder(state)
        
        # Shared processing
        shared_features = self.shared_layers(encoded_state)
        
        # Separate value estimates
        wealth_value = self.wealth_head(shared_features).squeeze(-1)
        goal_value = self.goal_head(shared_features).squeeze(-1)
        
        # Combine values with learnable weights
        combined_value = (
            torch.sigmoid(self.wealth_weight) * wealth_value +
            torch.sigmoid(self.goal_weight) * goal_value
        )
        
        return combined_value
    
    def get_component_values(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get separate component value estimates for analysis
        
        Args:
            state: Input state tensor
            
        Returns:
            Dictionary with wealth and goal value components
        """
        encoded_state = self.state_encoder(state)
        shared_features = self.shared_layers(encoded_state)
        
        wealth_value = self.wealth_head(shared_features).squeeze(-1)
        goal_value = self.goal_head(shared_features).squeeze(-1)
        combined_value = (
            torch.sigmoid(self.wealth_weight) * wealth_value +
            torch.sigmoid(self.goal_weight) * goal_value
        )
        
        return {
            'wealth_value': wealth_value,
            'goal_value': goal_value,
            'combined_value': combined_value,
            'wealth_weight': torch.sigmoid(self.wealth_weight),
            'goal_weight': torch.sigmoid(self.goal_weight)
        }
    
    def has_attention_encoder(self) -> bool:
        """Check if using per-feature attention encoder"""
        return isinstance(self.state_encoder, PerFeatureAttentionEncoder)
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from per-feature attention encoder
        
        Returns:
            attention_weights: (input_dim, input_dim) attention matrix if using per-feature attention,
                             None otherwise
        """
        if self.has_attention_encoder():
            return self.state_encoder.get_attention_weights()
        return None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from per-feature attention encoder
        
        Returns:
            Dictionary mapping feature names to importance scores if using per-feature attention,
            None otherwise
        """
        if self.has_attention_encoder():
            return self.state_encoder.get_feature_importance()
        return None
    
    def get_attention_summary(self) -> Optional[Dict[str, any]]:
        """
        Get comprehensive attention analysis from per-feature attention encoder
        
        Returns:
            Dictionary with attention patterns, regime indicators, etc. if using per-feature attention,
            None otherwise
        """
        if self.has_attention_encoder():
            return self.state_encoder.get_attention_summary()
        return None


class EnsembleValueNetwork(nn.Module):
    """
    Ensemble of value networks for improved robustness
    
    Combines multiple value networks to reduce estimation variance
    and improve training stability.
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64,
        encoder_type: str = "feature",
        num_networks: int = 3
    ):
        """
        Initialize ensemble value network
        
        Args:
            state_dim: Input state dimensionality
            hidden_dim: Hidden layer dimension
            encoder_type: Type of encoder to use
            num_networks: Number of networks in ensemble
        """
        super().__init__()
        
        self.num_networks = num_networks
        self.encoder_type = encoder_type
        
        # Create ensemble of value networks
        self.networks = nn.ModuleList([
            SentimentAwareValueNetwork(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                encoder_type=encoder_type,
                num_layers=2 + i % 2  # Vary architecture slightly
            )
            for i in range(num_networks)
        ])
        
        # Combination weights
        self.combination_weights = nn.Parameter(
            torch.ones(num_networks) / num_networks
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble
        
        Args:
            state: Input state tensor
            
        Returns:
            Weighted average of ensemble predictions
        """
        # Get predictions from all networks
        predictions = torch.stack([
            network(state) for network in self.networks
        ], dim=0)  # (num_networks, batch_size)
        
        # Compute weighted average
        weights = torch.softmax(self.combination_weights, dim=0)
        ensemble_value = torch.sum(
            weights.unsqueeze(1) * predictions, dim=0
        )
        
        return ensemble_value
    
    def get_individual_values(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get individual network predictions for analysis
        
        Args:
            state: Input state tensor
            
        Returns:
            Dictionary with individual and ensemble predictions
        """
        individual_predictions = [
            network(state) for network in self.networks
        ]
        
        ensemble_prediction = self.forward(state)
        weights = torch.softmax(self.combination_weights, dim=0)
        
        return {
            'individual_predictions': individual_predictions,
            'ensemble_prediction': ensemble_prediction,
            'network_weights': weights,
            'prediction_std': torch.std(torch.stack(individual_predictions, dim=0), dim=0)
        }
    
    def has_attention_encoder(self) -> bool:
        """Check if any network in ensemble uses per-feature attention encoder"""
        return any(isinstance(network.state_encoder, PerFeatureAttentionEncoder) for network in self.networks)
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from first per-feature attention encoder in ensemble
        
        Returns:
            attention_weights: (input_dim, input_dim) attention matrix from first attention network,
                             None if no attention networks in ensemble
        """
        for network in self.networks:
            if isinstance(network.state_encoder, PerFeatureAttentionEncoder):
                return network.state_encoder.get_attention_weights()
        return None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get aggregated feature importance from all per-feature attention networks in ensemble
        
        Returns:
            Dictionary mapping feature names to average importance scores across attention networks,
            None if no attention networks in ensemble
        """
        attention_networks = []
        for network in self.networks:
            if isinstance(network.state_encoder, PerFeatureAttentionEncoder):
                importance = network.state_encoder.get_feature_importance()
                if importance:
                    attention_networks.append(importance)
        
        if not attention_networks:
            return None
        
        # Average importance scores across ensemble
        aggregated_importance = {}
        feature_names = attention_networks[0].keys()
        for feature in feature_names:
            scores = [importance[feature] for importance in attention_networks]
            aggregated_importance[feature] = sum(scores) / len(scores)
        
        return aggregated_importance
    
    def get_attention_summary(self) -> Optional[Dict[str, any]]:
        """
        Get aggregated attention analysis from all per-feature attention networks in ensemble
        
        Returns:
            Dictionary with ensemble-averaged attention patterns, regime indicators, etc.,
            None if no attention networks in ensemble
        """
        attention_summaries = []
        for network in self.networks:
            if isinstance(network.state_encoder, PerFeatureAttentionEncoder):
                summary = network.state_encoder.get_attention_summary()
                if summary:
                    attention_summaries.append(summary)
        
        if not attention_summaries:
            return None
        
        # Aggregate attention summaries
        aggregated_summary = {
            'num_attention_networks': len(attention_summaries),
            'total_networks': len(self.networks),
            'attention_coverage': len(attention_summaries) / len(self.networks)
        }
        
        # Average numeric metrics
        if 'vix_total_attention' in attention_summaries[0]:
            vix_attentions = [s['vix_total_attention'] for s in attention_summaries]
            aggregated_summary['avg_vix_total_attention'] = sum(vix_attentions) / len(vix_attentions)
        
        if 'time_vs_wealth_ratio' in attention_summaries[0]:
            ratios = [s['time_vs_wealth_ratio'] for s in attention_summaries]
            aggregated_summary['avg_time_vs_wealth_ratio'] = sum(ratios) / len(ratios)
        
        # Consensus regime prediction (mode)
        if 'predicted_regime' in attention_summaries[0]:
            regimes = [s['predicted_regime'] for s in attention_summaries]
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            consensus_regime = max(regime_counts.items(), key=lambda x: x[1])
            aggregated_summary['consensus_regime'] = consensus_regime[0]
            aggregated_summary['regime_confidence'] = consensus_regime[1] / len(regimes)
        
        # Include feature importance from aggregated method
        aggregated_summary['feature_importance'] = self.get_feature_importance()
        
        return aggregated_summary


def create_sentiment_value_network(
    network_type: str = "standard",
    state_dim: int = 4,
    **kwargs
) -> nn.Module:
    """
    Factory function to create sentiment-aware value networks
    
    Args:
        network_type: Type of network ('standard', 'dual_head', 'ensemble')
        state_dim: Input state dimensionality
        **kwargs: Additional network parameters
        
    Returns:
        Initialized value network
    """
    if network_type == "standard":
        return SentimentAwareValueNetwork(state_dim=state_dim, **kwargs)
    elif network_type == "dual_head":
        return DualHeadValueNetwork(state_dim=state_dim, **kwargs)
    elif network_type == "ensemble":
        return EnsembleValueNetwork(state_dim=state_dim, **kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def test_sentiment_value_networks():
    """Test function for sentiment-aware value networks"""
    print("Testing sentiment-aware value networks...")
    
    batch_size = 32
    state_2d = torch.randn(batch_size, 2)
    state_4d = torch.randn(batch_size, 4)
    
    try:
        # Test standard sentiment-aware value network
        value_net = SentimentAwareValueNetwork(state_dim=4, hidden_dim=64)
        
        # Test forward pass
        values = value_net.forward(state_4d)
        assert values.shape == (batch_size,), f"Wrong values shape: {values.shape}"
        print("✓ Standard value network forward pass test passed")
        
        # Test single state input
        single_state = state_4d[0]  # Shape: (4,)
        single_value = value_net.get_value(single_state.unsqueeze(0))
        assert single_value.shape == (1,), f"Wrong single value shape: {single_value.shape}"
        print("✓ Single state input test passed")
        
        # Test dual-head value network
        dual_net = DualHeadValueNetwork(state_dim=4)
        
        dual_values = dual_net.forward(state_4d)
        assert dual_values.shape == (batch_size,), f"Wrong dual values shape: {dual_values.shape}"
        
        component_values = dual_net.get_component_values(state_4d)
        assert 'wealth_value' in component_values, "Missing wealth value component"
        assert 'goal_value' in component_values, "Missing goal value component"
        assert 'combined_value' in component_values, "Missing combined value"
        print("✓ Dual-head value network test passed")
        
        # Test ensemble value network
        ensemble_net = EnsembleValueNetwork(state_dim=4, num_networks=3)
        
        ensemble_values = ensemble_net.forward(state_4d)
        assert ensemble_values.shape == (batch_size,), f"Wrong ensemble values shape: {ensemble_values.shape}"
        
        individual_values = ensemble_net.get_individual_values(state_4d)
        assert 'individual_predictions' in individual_values, "Missing individual predictions"
        assert 'ensemble_prediction' in individual_values, "Missing ensemble prediction"
        assert len(individual_values['individual_predictions']) == 3, "Wrong number of individual predictions"
        print("✓ Ensemble value network test passed")
        
        # Test adaptive state dimensions
        value_net_2d = SentimentAwareValueNetwork(
            state_dim=2,
            encoder_type="simple",
            hidden_dim=64
        )
        
        values_2d = value_net_2d.forward(state_2d)
        assert values_2d.shape == (batch_size,), f"Wrong 2D values shape: {values_2d.shape}"
        print("✓ Adaptive state dimensions test passed")
        
        # Test different encoder types
        encoders_to_test = ["simple", "adaptive"]
        for encoder_type in encoders_to_test:
            test_net = SentimentAwareValueNetwork(
                state_dim=4,
                encoder_type=encoder_type,
                hidden_dim=64
            )
            test_values = test_net.forward(state_4d)
            assert test_values.shape == (batch_size,), f"Wrong values shape for {encoder_type}: {test_values.shape}"
        print("✓ Different encoder types test passed")
        
        # Test factory function
        networks = [
            create_sentiment_value_network("standard", state_dim=4),
            create_sentiment_value_network("dual_head", state_dim=4),
            create_sentiment_value_network("ensemble", state_dim=4)
        ]
        print("✓ Value network factory test passed")
        
        # Test gradient flow
        test_net = SentimentAwareValueNetwork(state_dim=4)
        test_state = torch.randn(10, 4, requires_grad=True)
        test_values = test_net(test_state)
        loss = test_values.mean()
        loss.backward()
        
        # Check that gradients exist
        assert test_state.grad is not None, "No gradients computed for input"
        print("✓ Gradient flow test passed")
        
        # Test per-feature attention value networks
        state_5d = torch.randn(batch_size, 5)  # [time, wealth, vix_level, vix_avg, vix_momentum]
        
        # Test standard value network with per-feature attention
        pfa_value_net = SentimentAwareValueNetwork(
            state_dim=5,
            encoder_type="per_feature_attention",
            hidden_dim=64
        )
        
        pfa_values = pfa_value_net.forward(state_5d)
        assert pfa_values.shape == (batch_size,), f"Wrong PFA values shape: {pfa_values.shape}"
        print("✓ Per-feature attention value network forward pass test passed")
        
        # Test dual-head value network with per-feature attention
        pfa_dual_net = DualHeadValueNetwork(
            state_dim=5,
            encoder_type="per_feature_attention"
        )
        
        pfa_dual_values = pfa_dual_net.forward(state_5d)
        assert pfa_dual_values.shape == (batch_size,), f"Wrong PFA dual values shape: {pfa_dual_values.shape}"
        
        # Test attention features in dual-head
        assert pfa_dual_net.has_attention_encoder(), "Dual-head network should detect per-feature attention encoder"
        
        dual_attention_weights = pfa_dual_net.get_attention_weights()
        assert dual_attention_weights is not None, "Dual-head network should return attention weights"
        assert dual_attention_weights.shape == (5, 5), f"Wrong dual-head attention weights shape: {dual_attention_weights.shape}"
        
        dual_feature_importance = pfa_dual_net.get_feature_importance()
        assert dual_feature_importance is not None, "Dual-head network should return feature importance"
        expected_features = ['time', 'wealth', 'vix_level', 'vix_avg', 'vix_momentum']
        assert all(feat in dual_feature_importance for feat in expected_features), "Missing features in dual-head importance"
        
        dual_attention_summary = pfa_dual_net.get_attention_summary()
        assert dual_attention_summary is not None, "Dual-head network should return attention summary"
        assert 'predicted_regime' in dual_attention_summary, "Missing regime prediction in dual-head"
        
        print("✓ Dual-head value network with per-feature attention test passed")
        
        # Test ensemble value network with per-feature attention
        pfa_ensemble_net = EnsembleValueNetwork(
            state_dim=5,
            encoder_type="per_feature_attention",
            num_networks=3
        )
        
        pfa_ensemble_values = pfa_ensemble_net.forward(state_5d)
        assert pfa_ensemble_values.shape == (batch_size,), f"Wrong PFA ensemble values shape: {pfa_ensemble_values.shape}"
        
        # Test attention features in ensemble
        assert pfa_ensemble_net.has_attention_encoder(), "Ensemble network should detect per-feature attention encoders"
        
        ensemble_attention_weights = pfa_ensemble_net.get_attention_weights()
        assert ensemble_attention_weights is not None, "Ensemble network should return attention weights"
        assert ensemble_attention_weights.shape == (5, 5), f"Wrong ensemble attention weights shape: {ensemble_attention_weights.shape}"
        
        ensemble_feature_importance = pfa_ensemble_net.get_feature_importance()
        assert ensemble_feature_importance is not None, "Ensemble network should return aggregated feature importance"
        assert all(feat in ensemble_feature_importance for feat in expected_features), "Missing features in ensemble importance"
        
        ensemble_attention_summary = pfa_ensemble_net.get_attention_summary()
        assert ensemble_attention_summary is not None, "Ensemble network should return aggregated attention summary"
        assert 'consensus_regime' in ensemble_attention_summary, "Missing consensus regime in ensemble"
        assert 'attention_coverage' in ensemble_attention_summary, "Missing attention coverage in ensemble"
        
        print("✓ Ensemble value network with per-feature attention test passed")
        
        # Test factory function with per-feature attention
        pfa_networks = [
            create_sentiment_value_network("standard", state_dim=5, encoder_type="per_feature_attention"),
            create_sentiment_value_network("dual_head", state_dim=5, encoder_type="per_feature_attention"),
            create_sentiment_value_network("ensemble", state_dim=5, encoder_type="per_feature_attention")
        ]
        
        # Verify all networks support attention
        for i, network in enumerate(pfa_networks):
            assert hasattr(network, 'has_attention_encoder'), f"Network {i} missing attention interface"
            
            # Run forward pass to generate attention
            _ = network(state_5d)
            
            assert network.has_attention_encoder(), f"Network {i} should have attention encoder"
            attention_weights = network.get_attention_weights()
            assert attention_weights is not None, f"Network {i} should return attention weights"
            
        print("✓ Per-feature attention value network factory test passed")
        
        print("All sentiment value network tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"✗ Sentiment value network test failed: {e}")
        return False


if __name__ == "__main__":
    test_sentiment_value_networks()