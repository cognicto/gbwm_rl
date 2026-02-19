"""Training configuration for GBWM RL project"""

from dataclasses import dataclass
from typing import Dict, Any
import torch


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration"""

    # PPO Hyperparameters (from paper)
    n_traj: int = 50000  # Total trajectories
    batch_size: int = 4800  # M: trajectories per batch
    learning_rate: float = 0.01  # η: initial learning rate
    clip_epsilon: float = 0.50  # ε: PPO clip parameter
    n_neurons: int = 64  # Neurons per hidden layer

    # Training settings
    ppo_epochs: int = 4  # Epochs per batch
    mini_batch_size: int = 256  # Mini-batch size for training
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    entropy_coeff: float = 0.01  # Entropy bonus coefficient
    value_loss_coeff: float = 0.5  # Value function loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Environment settings
    time_horizon: int = 16  # T: years
    num_goals: int = 4  # Number of goals
    num_portfolios: int = 15  # Portfolio choices
    initial_wealth_base: float = 12.0  # Base initial wealth
    wealth_scaling: float = 0.85  # Wealth scaling exponent

    # Device and reproducibility
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42

    # Logging
    log_interval: int = 10  # Log every N iterations
    save_interval: int = 50  # Save model every N iterations
    eval_interval: int = 20  # Evaluate every N iterations

    # Data mode configuration
    data_mode: str = "simulation"  # "simulation" or "historical"
    historical_data_path: str = "data/raw/market_data/"
    historical_start_date: str = "1970-01-01"
    historical_end_date: str = "2023-12-31"
    
    # Sentiment configuration
    sentiment_enabled: bool = False  # Enable VIX sentiment integration
    sentiment_start_date: str = "2015-01-01"  # Start date for sentiment data
    vix_weight: float = 1.0  # Weight for VIX sentiment component
    
    # Historical training parameters
    historical_validation_episodes: int = 1000  # Episodes for historical validation
    historical_augmentation: bool = True  # Enable data augmentation for historical mode
    
    def get_initial_wealth(self, num_goals: int) -> float:
        """Calculate initial wealth based on number of goals"""
        return self.initial_wealth_base * (num_goals ** self.wealth_scaling) * 10000
    
    def is_historical_mode(self) -> bool:
        """Check if configuration is set for historical data mode"""
        return self.data_mode == "historical"
    
    def get_experiment_suffix(self) -> str:
        """Get suffix for experiment name based on data mode"""
        if self.is_historical_mode():
            return f"_hist_{self.data_mode}"
        return f"_sim_{self.data_mode}"


@dataclass
class ModelConfig:
    """Neural network architecture configuration"""

    # Network architecture
    state_dim: int = 2  # [time, wealth] baseline, [time, wealth, vix_sentiment, vix_momentum] sentiment
    goal_action_dim: int = 2  # [skip, take]
    portfolio_action_dim: int = 15  # Number of portfolios
    hidden_dim: int = 64  # Hidden layer size
    n_hidden_layers: int = 2  # Number of hidden layers

    # Activation functions
    activation: str = "relu"  # relu, tanh, etc.
    output_activation: str = "softmax"  # For probability distributions
    
    def get_sentiment_state_dim(self) -> int:
        """Get state dimension for sentiment-aware models"""
        return 4  # [time, wealth, vix_sentiment, vix_momentum]


# Default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()