"""
Pure RL Value Network (Critic) with Advanced Architectures for GBWM

This module implements Pure RL value networks with the same architectural
options as Sentiment RL, but for 2D state [time, wealth] without VIX features.

Available architectures:
- Standard: encoder → value layers → scalar output
- Dual-head: separate wealth and goal value heads
- Ensemble: multiple value networks combined
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from .pure_rl_policy_network import create_encoder_2d


class PureRLValueNetwork(nn.Module):
    """
    Pure RL value network with advanced architecture options

    Estimates state values V(s) for 2D state [time, wealth]:
        State Encoder → Value Layers → Scalar Output

    This mirrors SentimentAwareValueNetwork but for 2D state.
    """

    def __init__(
        self,
        state_dim: int = 2,
        hidden_dim: int = 64,
        encoder_type: str = "simple",
        num_layers: int = 2,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0
    ):
        """
        Initialize Pure RL value network

        Args:
            state_dim: Input state dimensionality (default: 2 for [time, wealth])
            hidden_dim: Hidden layer dimension
            encoder_type: Type of encoder ('simple', 'attention')
            num_layers: Number of value layers after encoding
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0.0 to disable)
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.num_layers = num_layers

        # State encoder
        self.state_encoder = create_encoder_2d(
            encoder_type=encoder_type,
            input_dim=state_dim,
            hidden_dim=hidden_dim
        )

        encoder_output_dim = self.state_encoder.get_output_dim()

        # Value estimation layers
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

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for value function learning"""
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
        encoded_state = self.state_encoder(state)
        value_features = self.value_layers(encoded_state)
        values = self.value_head(value_features)
        values = values.squeeze(-1)

        return values

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Convenience method to get value estimates"""
        return self.forward(state)


class DualHeadValueNetwork2D(nn.Module):
    """
    Dual-head value network for 2D state

    Estimates two components of value:
    1. Wealth value: Expected future wealth accumulation
    2. Goal value: Expected future goal utilities

    This decomposition helps with interpretability and training stability.
    """

    def __init__(
        self,
        state_dim: int = 2,
        hidden_dim: int = 64,
        encoder_type: str = "simple"
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Shared state encoder
        self.state_encoder = create_encoder_2d(
            encoder_type=encoder_type,
            input_dim=state_dim,
            hidden_dim=hidden_dim
        )

        encoder_output_dim = self.state_encoder.get_output_dim()

        # Shared processing layers
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
        for module in [self.shared_layers, self.wealth_head, self.goal_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual-head value network

        Returns:
            Combined value estimates
        """
        encoded_state = self.state_encoder(state)
        shared_features = self.shared_layers(encoded_state)

        wealth_value = self.wealth_head(shared_features).squeeze(-1)
        goal_value = self.goal_head(shared_features).squeeze(-1)

        combined_value = (
            torch.sigmoid(self.wealth_weight) * wealth_value +
            torch.sigmoid(self.goal_weight) * goal_value
        )

        return combined_value

    def get_component_values(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get separate component value estimates for analysis"""
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


class EnsembleValueNetwork2D(nn.Module):
    """
    Ensemble of value networks for 2D state

    Combines multiple value networks to reduce estimation variance
    and improve training stability.
    """

    def __init__(
        self,
        state_dim: int = 2,
        hidden_dim: int = 64,
        encoder_type: str = "simple",
        num_networks: int = 3
    ):
        super().__init__()

        self.num_networks = num_networks

        # Create ensemble of value networks
        self.networks = nn.ModuleList([
            PureRLValueNetwork(
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

        Returns:
            Weighted average of ensemble predictions
        """
        predictions = torch.stack([
            network(state) for network in self.networks
        ], dim=0)  # (num_networks, batch_size)

        weights = torch.softmax(self.combination_weights, dim=0)
        ensemble_value = torch.sum(
            weights.unsqueeze(1) * predictions, dim=0
        )

        return ensemble_value

    def get_individual_values(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get individual network predictions for analysis"""
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


def create_pure_rl_value_network(
    value_type: str = "standard",
    state_dim: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create Pure RL value networks

    Args:
        value_type: Type of network ('standard', 'dual_head', 'ensemble')
        state_dim: Input state dimensionality (default: 2)
        **kwargs: Additional network parameters

    Returns:
        Initialized value network
    """
    if value_type == "standard":
        return PureRLValueNetwork(state_dim=state_dim, **kwargs)
    elif value_type == "dual_head":
        # Filter to only supported kwargs
        supported_kwargs = {'hidden_dim', 'encoder_type'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}
        return DualHeadValueNetwork2D(state_dim=state_dim, **filtered_kwargs)
    elif value_type == "ensemble":
        # Filter to only supported kwargs
        supported_kwargs = {'hidden_dim', 'encoder_type', 'num_networks'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}
        return EnsembleValueNetwork2D(state_dim=state_dim, **filtered_kwargs)
    else:
        raise ValueError(f"Unknown value type: {value_type}. Use 'standard', 'dual_head', or 'ensemble'.")


def test_pure_rl_value_networks():
    """Test function for Pure RL value networks"""
    print("Testing Pure RL value networks (2D state)...")

    batch_size = 32
    state_2d = torch.randn(batch_size, 2)  # [time, wealth]

    try:
        # Test standard value network with simple encoder
        value_simple = PureRLValueNetwork(
            state_dim=2,
            hidden_dim=64,
            encoder_type="simple"
        )

        values = value_simple.forward(state_2d)
        assert values.shape == (batch_size,), f"Wrong values shape: {values.shape}"
        print("✓ Standard value network (simple encoder) test passed")

        # Test standard value network with attention encoder
        value_attention = PureRLValueNetwork(
            state_dim=2,
            hidden_dim=64,
            encoder_type="attention"
        )

        values_att = value_attention.forward(state_2d)
        assert values_att.shape == (batch_size,), f"Wrong attention values shape: {values_att.shape}"
        print("✓ Standard value network (attention encoder) test passed")

        # Test dual-head value network
        value_dual = DualHeadValueNetwork2D(state_dim=2)

        dual_values = value_dual.forward(state_2d)
        assert dual_values.shape == (batch_size,), f"Wrong dual values shape: {dual_values.shape}"

        component_values = value_dual.get_component_values(state_2d)
        assert 'wealth_value' in component_values, "Missing wealth value component"
        assert 'goal_value' in component_values, "Missing goal value component"
        print("✓ Dual-head value network test passed")

        # Test ensemble value network
        value_ensemble = EnsembleValueNetwork2D(state_dim=2, num_networks=3)

        ensemble_values = value_ensemble.forward(state_2d)
        assert ensemble_values.shape == (batch_size,), f"Wrong ensemble values shape: {ensemble_values.shape}"

        individual_values = value_ensemble.get_individual_values(state_2d)
        assert len(individual_values['individual_predictions']) == 3, "Wrong number of individual predictions"
        print("✓ Ensemble value network test passed")

        # Test single state input
        single_state = state_2d[0].unsqueeze(0)  # Shape: (1, 2)
        single_value = value_simple.get_value(single_state)
        assert single_value.shape == (1,), f"Wrong single value shape: {single_value.shape}"
        print("✓ Single state input test passed")

        # Test factory function
        networks = [
            create_pure_rl_value_network("standard", encoder_type="simple"),
            create_pure_rl_value_network("standard", encoder_type="attention"),
            create_pure_rl_value_network("dual_head"),
            create_pure_rl_value_network("ensemble")
        ]
        print("✓ Value network factory test passed")

        # Test gradient flow
        test_state = torch.randn(10, 2, requires_grad=True)
        test_values = value_simple(test_state)
        loss = test_values.mean()
        loss.backward()

        assert test_state.grad is not None, "No gradients computed for input"
        print("✓ Gradient flow test passed")

        print("\nAll Pure RL value network tests passed! ✓")
        return True

    except Exception as e:
        print(f"✗ Pure RL value network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_pure_rl_value_networks()