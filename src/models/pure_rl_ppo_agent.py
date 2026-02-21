"""
Pure RL PPO Agent with Advanced Architectures for GBWM

Implements PPO with configurable policy and value network architectures
for 2D state [time, wealth] without VIX features.

Supports:
- Policy types: standard, hierarchical
- Value types: standard, dual_head, ensemble
- Encoder types: simple, attention
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque
from dataclasses import dataclass

from .pure_rl_policy_network import create_pure_rl_policy, PureRLPolicyNetwork, HierarchicalPolicyNetwork2D
from .pure_rl_value_network import create_pure_rl_value_network, PureRLValueNetwork
from src.utils.data_utils import compute_gae


@dataclass
class PureRLAgentConfig:
    """Configuration for Pure RL PPO Agent with advanced architectures"""
    # Architecture options
    policy_type: str = "standard"       # 'standard', 'hierarchical'
    value_type: str = "standard"        # 'standard', 'dual_head', 'ensemble'
    encoder_type: str = "simple"        # 'simple', 'attention'

    # Network parameters
    hidden_dim: int = 64
    use_batch_norm: bool = False
    dropout_rate: float = 0.0

    # Training parameters
    learning_rate: float = 0.01
    clip_epsilon: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # Environment parameters
    state_dim: int = 2
    num_portfolios: int = 15
    time_horizon: int = 16  # Annual steps
    batch_size: int = 4800  # Episodes per update
    n_traj: int = 48000     # Total trajectories

    # Other
    device: str = "cpu"
    random_seed: int = 42
    log_interval: int = 1


class PureRLPPOAgent:
    """
    Pure RL PPO Agent with configurable advanced architectures

    Supports different policy types (standard, hierarchical),
    value types (standard, dual_head, ensemble), and
    encoder types (simple, attention).
    """

    def __init__(
        self,
        env,
        config: PureRLAgentConfig = None,
        device: str = None
    ):
        """
        Initialize Pure RL PPO Agent

        Args:
            env: GBWM environment
            config: Agent configuration
            device: Device to run on ('cuda' or 'cpu')
        """
        self.env = env
        self.config = config or PureRLAgentConfig()
        self.device = device or self.config.device

        # Log architecture configuration
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing PureRLPPOAgent:")
        self.logger.info(f"  Policy type: {self.config.policy_type}")
        self.logger.info(f"  Value type: {self.config.value_type}")
        self.logger.info(f"  Encoder type: {self.config.encoder_type}")

        # Initialize policy network
        self.policy_net = create_pure_rl_policy(
            policy_type=self.config.policy_type,
            state_dim=self.config.state_dim,
            num_portfolios=self.config.num_portfolios,
            hidden_dim=self.config.hidden_dim,
            encoder_type=self.config.encoder_type,
            use_batch_norm=self.config.use_batch_norm,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)

        # Initialize value network
        self.value_net = create_pure_rl_value_network(
            value_type=self.config.value_type,
            state_dim=self.config.state_dim,
            hidden_dim=self.config.hidden_dim,
            encoder_type=self.config.encoder_type
        ).to(self.device)

        # Count parameters
        policy_params = sum(p.numel() for p in self.policy_net.parameters())
        value_params = sum(p.numel() for p in self.value_net.parameters())
        self.logger.info(f"  Policy parameters: {policy_params:,}")
        self.logger.info(f"  Value parameters: {value_params:,}")

        # Initialize optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=self.config.learning_rate
        )

        # Learning rate schedulers
        total_iterations = self.config.n_traj // self.config.batch_size
        self.policy_scheduler = optim.lr_scheduler.LinearLR(
            self.policy_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_iterations
        )
        self.value_scheduler = optim.lr_scheduler.LinearLR(
            self.value_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_iterations
        )

        # Training metrics
        self.training_metrics = {
            'policy_losses': deque(maxlen=100),
            'value_losses': deque(maxlen=100),
            'total_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'advantages': deque(maxlen=1000)
        }

        # Training state
        self.total_timesteps = 0
        self.total_episodes = 0
        self.iteration = 0

    def collect_trajectories(self, num_trajectories: int) -> Dict[str, torch.Tensor]:
        """
        Collect batch of trajectories using current policy

        Args:
            num_trajectories: Number of trajectories to collect

        Returns:
            Dictionary containing collected data
        """
        trajectories = []

        self.policy_net.eval()
        self.value_net.eval()

        with torch.no_grad():
            for traj_idx in range(num_trajectories):
                trajectory = []

                # Reset environment
                obs, _ = self.env.reset()

                for step in range(self.config.time_horizon):
                    # Convert observation to tensor
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                    # Get action and log probability
                    actions, log_probs = self.policy_net.get_action_and_log_prob(state_tensor)

                    # Get value estimate
                    values = self.value_net(state_tensor)

                    # Execute action in environment
                    action_np = actions.squeeze(0).cpu().numpy()
                    next_obs, reward, terminated, truncated, info = self.env.step(action_np)

                    # Store step data
                    step_data = {
                        'state': obs.copy(),
                        'action': actions.squeeze(0).cpu(),
                        'reward': reward,
                        'log_prob': log_probs.squeeze(0).cpu(),
                        'value': values.squeeze(0).cpu(),
                        'done': terminated or truncated
                    }
                    trajectory.append(step_data)

                    obs = next_obs
                    self.total_timesteps += 1

                    if terminated or truncated:
                        break

                trajectories.append(trajectory)
                self.total_episodes += 1

                # Track episode metrics
                episode_reward = sum(step['reward'] for step in trajectory)
                self.training_metrics['total_rewards'].append(episode_reward)
                self.training_metrics['episode_lengths'].append(len(trajectory))

        # Convert trajectories to tensors
        return self._process_trajectories(trajectories)

    def _process_trajectories(self, trajectories: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """Process collected trajectories into training format"""
        all_states = []
        all_actions = []
        all_rewards = []
        all_log_probs = []
        all_values = []
        all_dones = []

        for trajectory in trajectories:
            for step in trajectory:
                all_states.append(step['state'])
                all_actions.append(step['action'])
                all_rewards.append(step['reward'])
                all_log_probs.append(step['log_prob'])
                all_values.append(step['value'])
                all_dones.append(step['done'])

        # Convert to tensors
        states = torch.FloatTensor(np.array(all_states)).to(self.device)
        actions = torch.stack(all_actions).to(self.device)
        rewards = torch.FloatTensor(all_rewards).to(self.device)
        old_log_probs = torch.stack(all_log_probs).to(self.device)
        values = torch.stack(all_values).to(self.device)
        dones = torch.tensor(all_dones, dtype=torch.bool).to(self.device)

        # Compute advantages and returns using GAE
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.training_metrics['advantages'].extend(advantages.cpu().numpy())

        return {
            'states': states,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns,
            'old_values': values
        }

    def update_policy(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update policy and value networks using PPO

        Args:
            batch_data: Batch of trajectory data

        Returns:
            Dictionary of training metrics
        """
        self.policy_net.train()
        self.value_net.train()

        policy_losses = []
        value_losses = []

        batch_size = batch_data['states'].shape[0]

        # PPO update epochs
        for epoch in range(self.config.ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, batch_size)
                mb_indices = indices[start:end]

                mb_states = batch_data['states'][mb_indices]
                mb_actions = batch_data['actions'][mb_indices]
                mb_old_log_probs = batch_data['old_log_probs'][mb_indices]
                mb_advantages = batch_data['advantages'][mb_indices]
                mb_returns = batch_data['returns'][mb_indices]

                # === POLICY UPDATE ===
                new_log_probs, entropy = self.policy_net.evaluate_actions(mb_states, mb_actions)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                ) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -self.config.entropy_coeff * entropy.mean()
                total_policy_loss = policy_loss + entropy_loss

                self.policy_optimizer.zero_grad()
                total_policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()

                policy_losses.append(total_policy_loss.item())

                # === VALUE UPDATE ===
                new_values = self.value_net(mb_states)
                value_loss = nn.functional.mse_loss(new_values, mb_returns)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
                self.value_optimizer.step()

                value_losses.append(value_loss.item())

        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)

        self.training_metrics['policy_losses'].append(avg_policy_loss)
        self.training_metrics['value_losses'].append(avg_value_loss)

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'mean_advantage': batch_data['advantages'].mean().item(),
            'mean_return': batch_data['returns'].mean().item()
        }

    def train_iteration(self) -> Dict[str, float]:
        """
        Single training iteration: collect data + update networks

        Returns:
            Training metrics for this iteration
        """
        batch_data = self.collect_trajectories(self.config.batch_size)
        update_metrics = self.update_policy(batch_data)

        self.policy_scheduler.step()
        self.value_scheduler.step()

        self.iteration += 1

        metrics = {
            'iteration': self.iteration,
            'total_timesteps': self.total_timesteps,
            'total_episodes': self.total_episodes,
            'mean_episode_reward': np.mean(list(self.training_metrics['total_rewards'])),
            'mean_episode_length': np.mean(list(self.training_metrics['episode_lengths'])),
            'policy_lr': self.policy_scheduler.get_last_lr()[0],
            'value_lr': self.value_scheduler.get_last_lr()[0],
            **update_metrics
        }

        return metrics

    def train(self, num_iterations: int = 10) -> List[Dict[str, float]]:
        """
        Complete training loop

        Args:
            num_iterations: Number of training iterations

        Returns:
            List of training metrics for each iteration
        """
        self.logger.info(f"Starting training for {num_iterations} iterations")
        self.logger.info(f"Batch size: {self.config.batch_size}")

        training_history = []

        for iteration in range(num_iterations):
            metrics = self.train_iteration()
            training_history.append(metrics)

            if iteration % self.config.log_interval == 0:
                self.logger.info(
                    f"Iter {iteration}: "
                    f"Reward={metrics['mean_episode_reward']:.2f}, "
                    f"PolicyLoss={metrics['policy_loss']:.4f}, "
                    f"ValueLoss={metrics['value_loss']:.4f}"
                )

        self.logger.info("Training completed!")
        return training_history

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action for given observation

        Args:
            observation: Environment observation [time, wealth]
            deterministic: Whether to use deterministic policy

        Returns:
            Action array [goal_action, portfolio_action]
        """
        self.policy_net.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            actions, _ = self.policy_net.get_action_and_log_prob(
                state_tensor,
                deterministic=deterministic
            )
            return actions.squeeze(0).cpu().numpy()

    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the current architecture"""
        return {
            'policy_type': self.config.policy_type,
            'value_type': self.config.value_type,
            'encoder_type': self.config.encoder_type,
            'hidden_dim': self.config.hidden_dim,
            'policy_params': sum(p.numel() for p in self.policy_net.parameters()),
            'value_params': sum(p.numel() for p in self.value_net.parameters()),
            'total_params': sum(p.numel() for p in self.policy_net.parameters()) +
                           sum(p.numel() for p in self.value_net.parameters())
        }

    def save(self, filepath: str):
        """Save agent state"""
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': {
                'policy_type': self.config.policy_type,
                'value_type': self.config.value_type,
                'encoder_type': self.config.encoder_type,
                'hidden_dim': self.config.hidden_dim,
                'state_dim': self.config.state_dim,
                'num_portfolios': self.config.num_portfolios,
                'learning_rate': self.config.learning_rate,
                'clip_epsilon': self.config.clip_epsilon,
            },
            'training_metrics': {k: list(v) for k, v in self.training_metrics.items()},
            'total_timesteps': self.total_timesteps,
            'iteration': self.iteration
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.iteration = checkpoint.get('iteration', 0)

        if 'training_metrics' in checkpoint:
            for key, value in checkpoint['training_metrics'].items():
                if key in self.training_metrics:
                    self.training_metrics[key].extend(value)

        self.logger.info(f"Agent loaded from {filepath}")


def test_pure_rl_ppo_agent():
    """Test function for Pure RL PPO Agent"""
    print("Testing Pure RL PPO Agent...")

    # Create mock environment
    class MockEnv:
        def reset(self):
            return np.array([0.0, 1.0]), {}

        def step(self, action):
            return np.array([0.1, 1.05]), 1.0, False, False, {}

    try:
        # Test with different configurations
        configs = [
            PureRLAgentConfig(policy_type="standard", value_type="standard", encoder_type="simple"),
            PureRLAgentConfig(policy_type="standard", value_type="standard", encoder_type="attention"),
            PureRLAgentConfig(policy_type="hierarchical", value_type="dual_head", encoder_type="simple"),
            PureRLAgentConfig(policy_type="standard", value_type="ensemble", encoder_type="simple"),
        ]

        for i, config in enumerate(configs):
            print(f"\nTesting config {i+1}: policy={config.policy_type}, value={config.value_type}, encoder={config.encoder_type}")

            env = MockEnv()
            agent = PureRLPPOAgent(env, config=config)

            # Test prediction
            obs = np.array([0.0, 1.0])
            action = agent.predict(obs)
            assert action.shape == (2,), f"Wrong action shape: {action.shape}"
            print(f"  ✓ Prediction test passed")

            # Test architecture info
            info = agent.get_architecture_info()
            assert 'total_params' in info, "Missing total_params in architecture info"
            print(f"  ✓ Architecture info: {info['total_params']:,} total parameters")

        print("\n All Pure RL PPO Agent tests passed! ✓")
        return True

    except Exception as e:
        print(f"✗ Pure RL PPO Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_pure_rl_ppo_agent()
