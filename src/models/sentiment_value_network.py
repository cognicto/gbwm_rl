"""
Sentiment-Aware Value Network (Critic) for GBWM

This module implements sentiment-aware critic networks that estimate
state values considering market sentiment information.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any

from .feature_encoders import create_encoder


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
            encoder_type: Type of encoder ('feature', 'simple', 'adaptive', 'attention')
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
        
        print("All sentiment value network tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"✗ Sentiment value network test failed: {e}")
        return False


if __name__ == "__main__":
    test_sentiment_value_networks()