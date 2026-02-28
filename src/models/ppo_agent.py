"""
Proximal Policy Optimization (PPO) Agent for GBWM

Implements the complete PPO algorithm with custom multi-discrete action space handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque

from .policy_network import PolicyNetwork
from .value_network import ValueNetwork
from config.training_config import TrainingConfig, DEFAULT_TRAINING_CONFIG
from src.utils.data_utils import compute_gae


class PPOAgent:
    """
    PPO Agent for Goals-Based Wealth Management

    Implements the complete training loop with:
    - Multi-discrete action space handling
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Learning rate scheduling
    """

    def __init__(self,
                 env,
                 config: TrainingConfig = None,
                 device: str = None):
        """
        Initialize PPO Agent

        Args:
            env: GBWM environment
            config: Training configuration
            device: Device to run on ('cuda' or 'cpu')
        """
        self.env = env
        self.config = config or DEFAULT_TRAINING_CONFIG
        self.device = device or self.config.device

        # Initialize networks
        self.policy_net = PolicyNetwork().to(self.device)
        self.value_net = ValueNetwork().to(self.device)

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
        self.policy_scheduler = optim.lr_scheduler.LinearLR(
            self.policy_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.config.n_traj // self.config.batch_size
        )
        self.value_scheduler = optim.lr_scheduler.LinearLR(
            self.value_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.config.n_traj // self.config.batch_size
        )

        # Training metrics
        self.training_metrics = {
            'policy_losses': deque(maxlen=100),
            'value_losses': deque(maxlen=100),
            'total_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'advantages': deque(maxlen=1000)
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

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
        """
        Process collected trajectories into training format

        Args:
            trajectories: List of trajectory data

        Returns:
            Processed trajectory data as tensors
        """
        # Flatten all trajectory data
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
        dones = torch.tensor(all_dones, dtype=torch.bool).to(self.device)  # Explicitly set dtype

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

        # Store advantage statistics
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

        # Get batch size
        batch_size = batch_data['states'].shape[0]

        # PPO update epochs
        for epoch in range(self.config.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(batch_size, device=self.device)

            # Mini-batch updates
            for start in range(0, batch_size, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, batch_size)
                mb_indices = indices[start:end]

                # Extract mini-batch
                mb_states = batch_data['states'][mb_indices]
                mb_actions = batch_data['actions'][mb_indices]
                mb_old_log_probs = batch_data['old_log_probs'][mb_indices]
                mb_advantages = batch_data['advantages'][mb_indices]
                mb_returns = batch_data['returns'][mb_indices]

                # === POLICY UPDATE ===

                # Get current policy outputs
                new_log_probs, entropy = self.policy_net.evaluate_actions(mb_states, mb_actions)

                # Compute ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # PPO clipped objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                ) * mb_advantages

                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy_loss = -self.config.entropy_coeff * entropy.mean()

                # Total policy loss
                total_policy_loss = policy_loss + entropy_loss

                # Update policy
                self.policy_optimizer.zero_grad()
                total_policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()

                policy_losses.append(total_policy_loss.item())

                # === VALUE UPDATE ===

                # Get current value estimates
                new_values = self.value_net(mb_states)

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(new_values, mb_returns)

                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
                self.value_optimizer.step()

                value_losses.append(value_loss.item())

        # Store training metrics
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
        # Collect trajectories
        batch_data = self.collect_trajectories(self.config.batch_size)

        # Update networks
        update_metrics = self.update_policy(batch_data)

        # Update learning rates
        self.policy_scheduler.step()
        self.value_scheduler.step()

        # Increment iteration counter
        self.iteration += 1

        # Combine metrics
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

    def train(self, total_timesteps: int = None) -> List[Dict[str, float]]:
        """
        Complete training loop

        Args:
            total_timesteps: Total timesteps to train for

        Returns:
            List of training metrics for each iteration
        """
        if total_timesteps is None:
            total_timesteps = self.config.n_traj * self.config.time_horizon

        total_iterations = total_timesteps // (self.config.batch_size * self.config.time_horizon)

        self.logger.info(f"Starting training for {total_iterations} iterations")
        self.logger.info(f"Total timesteps: {total_timesteps}")
        self.logger.info(f"Batch size: {self.config.batch_size}")

        training_history = []

        for iteration in range(total_iterations):
            # Training iteration
            metrics = self.train_iteration()
            training_history.append(metrics)

            # Logging
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
            observation: Environment observation
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

    def save(self, filepath: str):
        """Save agent state with safe serialization"""
        # Convert config to dict to avoid serialization issues
        config_dict = {
            'n_traj': self.config.n_traj,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'clip_epsilon': self.config.clip_epsilon,
            'n_neurons': self.config.n_neurons,
            'ppo_epochs': self.config.ppo_epochs,
            'mini_batch_size': self.config.mini_batch_size,
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda,
            'entropy_coeff': self.config.entropy_coeff,
            'value_loss_coeff': self.config.value_loss_coeff,
            'max_grad_norm': self.config.max_grad_norm,
            'time_horizon': self.config.time_horizon,
            'num_goals': self.config.num_goals,
            'num_portfolios': self.config.num_portfolios,
            'initial_wealth_base': self.config.initial_wealth_base,
            'wealth_scaling': self.config.wealth_scaling,
            'device': str(self.config.device),
            'random_seed': self.config.random_seed,
            'log_interval': self.config.log_interval,
            'save_interval': self.config.save_interval,
            'eval_interval': self.config.eval_interval
        }

        # Convert training metrics to serializable format
        training_metrics_dict = {}
        for key, value in self.training_metrics.items():
            training_metrics_dict[key] = list(value)

        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config_dict': config_dict,  # Save as dict instead of object
            'training_metrics': training_metrics_dict,
            'total_timesteps': self.total_timesteps,
            'iteration': self.iteration
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state with safe deserialization"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Load network weights
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

        # Load training state
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.iteration = checkpoint.get('iteration', 0)

        # Load training metrics if available
        if 'training_metrics' in checkpoint:
            for key, value in checkpoint['training_metrics'].items():
                if key in self.training_metrics:
                    self.training_metrics[key].extend(value)

        self.logger.info(f"Agent loaded from {filepath}")