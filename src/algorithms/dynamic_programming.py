"""
Dynamic Programming Algorithm for Goals-Based Wealth Management

Implementation of the GBWM DP algorithm as described in:
"Dynamic Portfolio Allocation in Goals-Based Wealth Management" 
by Das, Ostrov, Radhakrishnan, and Srivastav (2019)
"""

import numpy as np
import time
from typing import Tuple, Dict, Optional, List
from scipy import stats
import logging
from dataclasses import dataclass

from config.environment_config import (
    EnvironmentConfig, DEFAULT_ENV_CONFIG,
    ExtendedGoalConfig, GoalDefinition, GoalOption
)


@dataclass
class DPConfig:
    """Configuration for Dynamic Programming algorithm"""
    # Base case parameters from paper
    initial_wealth: float = 100000.0  # W(0) = $100k
    goal_wealth: float = 200000.0     # G = $200k  
    time_horizon: int = 10            # T = 10 years
    
    # Portfolio parameters (efficient frontier bounds)
    mu_min: float = 0.0526           # Minimum expected return
    mu_max: float = 0.0886           # Maximum expected return  
    sigma_min: float = 0.0374        # Minimum volatility
    sigma_max: float = 0.1954        # Maximum volatility
    
    # Efficient frontier parameters calculated from paper's market data
    # Based on Table 1: US Bonds, International Stocks, US Stocks (1998-2017)
    eff_frontier_a: float = None     # Will be calculated from market data
    eff_frontier_b: float = None     # Will be calculated from market data  
    eff_frontier_c: float = None     # Will be calculated from market data
    
    # Algorithm parameters
    num_portfolios: int = 15         # m = number of portfolio choices
    grid_density: float = 1.5        # ρ_grid = wealth grid density (reduced for speed)
    
    # Cash flows (empty for base case)
    cash_flows: Dict[int, float] = None
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        """Calculate efficient frontier parameters from paper's market data"""
        if self.eff_frontier_a is None:
            self._calculate_efficient_frontier_params()
    
    def _calculate_efficient_frontier_params(self):
        """
        Calculate efficient frontier coefficients from paper's market data
        Based on Table 1 from Das et al. (2019) - January 1998 to December 2017
        """
        # Mean returns from Table 1
        mu = np.array([0.0493, 0.0770, 0.0886])  # [US Bonds, Intl Stocks, US Stocks]
        
        # Covariance matrix from Table 1  
        Sigma = np.array([
            [ 0.0017, -0.0017, -0.0021],
            [-0.0017,  0.0396,  0.0309], 
            [-0.0021,  0.0309,  0.0392]
        ])
        
        # Vector of ones
        ones = np.ones(3)
        
        # Calculate scalars k, l, p from paper's formulas
        Sigma_inv = np.linalg.inv(Sigma)
        k = mu.T @ Sigma_inv @ ones
        l = mu.T @ Sigma_inv @ mu  
        p = ones.T @ Sigma_inv @ ones
        
        # Calculate g and h vectors
        denominator = l * p - k**2
        g = (l * Sigma_inv @ ones - k * Sigma_inv @ mu) / denominator
        h = (p * Sigma_inv @ mu - k * Sigma_inv @ ones) / denominator
        
        # Calculate efficient frontier coefficients: σ = √(aμ² + bμ + c)
        self.eff_frontier_a = h.T @ Sigma @ h
        self.eff_frontier_b = 2 * g.T @ Sigma @ h  
        self.eff_frontier_c = g.T @ Sigma @ g
        
        # Verify bounds match expected values
        mu_test_min = 0.0526  # Should give σ ≈ 0.0374
        mu_test_max = 0.0886  # Should give σ ≈ 0.1954
        
        sigma_min_calc = np.sqrt(self.eff_frontier_a * mu_test_min**2 + 
                                self.eff_frontier_b * mu_test_min + 
                                self.eff_frontier_c)
        sigma_max_calc = np.sqrt(self.eff_frontier_a * mu_test_max**2 + 
                                self.eff_frontier_b * mu_test_max + 
                                self.eff_frontier_c)
        
        # Update bounds based on calculated values
        self.sigma_min = sigma_min_calc
        self.sigma_max = sigma_max_calc


class GBWMDynamicProgramming:
    """
    Goals-Based Wealth Management Dynamic Programming Algorithm
    
    Solves the optimization problem:
    max P[W(T) ≥ G] over all portfolio allocation strategies
    
    Uses backward recursion with the Bellman equation:
    V(Wi, t) = max[μ] Σ[j] V(Wj, t+1) × P(Wj at t+1 | Wi at t, portfolio μ)
    """
    
    def __init__(self, config: DPConfig = None):
        """Initialize the DP algorithm with configuration"""
        self.config = config or DPConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        np.random.seed(self.config.random_seed)
        
        # Initialize results storage first
        self.value_function = None  # V(Wi, t)
        self.policy = None          # μ*(Wi, t)
        self.wealth_grid = None     # W0, W1, ..., Wimax
        self.solve_time = None
        
        # Initialize components in correct order
        self._setup_efficient_frontier()
        self._setup_wealth_grid()
        
        self.logger.info("GBWM Dynamic Programming algorithm initialized")
    
    def _setup_efficient_frontier(self):
        """Setup the efficient frontier portfolios"""
        # Create m equally spaced μ values
        self.mu_array = np.linspace(self.config.mu_min, self.config.mu_max, self.config.num_portfolios)
        
        # Calculate corresponding σ values using efficient frontier equation
        # σ = √(aμ² + bμ + c)
        mu_squared = self.mu_array ** 2
        variance = (self.config.eff_frontier_a * mu_squared + 
                   self.config.eff_frontier_b * self.mu_array + 
                   self.config.eff_frontier_c)
        self.sigma_array = np.sqrt(np.maximum(variance, 0.0001))  # Ensure positive variance
        
        self.logger.info(f"Efficient frontier: {self.config.num_portfolios} portfolios")
        self.logger.info(f"μ range: [{self.config.mu_min:.4f}, {self.config.mu_max:.4f}]")
        self.logger.info(f"σ range: [{self.sigma_array.min():.4f}, {self.sigma_array.max():.4f}]")
    
    def _compute_wealth_bounds(self) -> Tuple[float, float]:
        """Compute extreme wealth bounds using worst/best case scenarios"""
        W0 = self.config.initial_wealth
        T = self.config.time_horizon
        
        # Worst case: minimum return + bad luck (-3σ)
        mu_worst = self.config.mu_min
        sigma_worst = self.config.sigma_max
        
        # Best case: maximum return + good luck (+3σ)  
        mu_best = self.config.mu_max
        sigma_best = self.config.sigma_max
        
        # Account for cash flows if present
        cash_flow_impact = 0.0
        if self.config.cash_flows:
            # Simple approximation - sum all cash flows
            cash_flow_impact = sum(self.config.cash_flows.values())
        
        # Extreme scenarios using geometric Brownian motion
        W_min = W0 * np.exp((mu_worst - 0.5 * sigma_worst**2) * T - 3 * sigma_worst * np.sqrt(T))
        W_max = W0 * np.exp((mu_best - 0.5 * sigma_best**2) * T + 3 * sigma_best * np.sqrt(T))
        
        # Add cash flow impact
        W_min = max(1.0, W_min + cash_flow_impact)  # Minimum $1 to avoid log issues
        W_max = W_max + abs(cash_flow_impact)
        
        return W_min, W_max
    
    def _setup_wealth_grid(self):
        """Create logarithmically-spaced wealth grid"""
        W_min, W_max = self._compute_wealth_bounds()
        
        # Logarithmic spacing
        ln_W_min = np.log(W_min)
        ln_W_max = np.log(W_max)
        
        # Grid size based on density parameter
        # Number of grid points per σ_min unit
        grid_span = ln_W_max - ln_W_min
        self.i_max = int(np.ceil(grid_span * self.config.grid_density / self.config.sigma_min))
        
        # Create logarithmically spaced grid
        ln_W_grid = np.linspace(ln_W_min, ln_W_max, self.i_max + 1)
        self.wealth_grid = np.exp(ln_W_grid)
        
        # Ensure initial wealth is in grid (find closest and replace)
        closest_idx = np.argmin(np.abs(self.wealth_grid - self.config.initial_wealth))
        self.wealth_grid[closest_idx] = self.config.initial_wealth
        self.initial_wealth_idx = closest_idx
        
        self.logger.info(f"Wealth grid: {len(self.wealth_grid)} points")
        self.logger.info(f"Range: [${self.wealth_grid.min():,.0f}, ${self.wealth_grid.max():,.0f}]")
        self.logger.info(f"Initial wealth at index: {self.initial_wealth_idx}")
    
    def _get_cash_flow(self, t: int) -> float:
        """Get cash flow at time t"""
        if self.config.cash_flows and t in self.config.cash_flows:
            return self.config.cash_flows[t]
        return 0.0
    
    def _compute_transition_probabilities(self, Wi: float, mu: float, sigma: float) -> np.ndarray:
        """
        Compute transition probabilities P(Wj at t+1 | Wi at t, portfolio μ)
        
        Uses geometric Brownian motion:
        W(t+1) = W(t) * exp((μ - σ²/2) + σ*Z) where Z ~ N(0,1)
        
        Returns normalized probability vector for all wealth grid points
        """
        # Add cash flow to current wealth  
        Wi_after_cashflow = Wi + self._get_cash_flow(0)  # Simplified - no time-dependent flows in base case
        
        if Wi_after_cashflow <= 0:
            # If bankrupt, stay at zero wealth
            probs = np.zeros(len(self.wealth_grid))
            zero_idx = np.argmin(np.abs(self.wealth_grid))
            probs[zero_idx] = 1.0
            return probs
        
        # Use discrete approximation for more stable computation
        probs = np.zeros(len(self.wealth_grid))
        
        # For each grid point, compute the probability of transitioning to it
        # using the discretized normal distribution
        for j, Wj in enumerate(self.wealth_grid):
            if Wj > 0:
                # Required log return to reach Wj from Wi
                log_return = np.log(Wj / Wi_after_cashflow)
                
                # Calculate probability using normal distribution
                # Log return ~ N(mu - sigma^2/2, sigma^2)
                mean_log_return = mu - 0.5 * sigma**2
                
                # Probability density
                z_score = (log_return - mean_log_return) / sigma
                
                # Use normal CDF for interval probability (more stable)
                if j == 0:
                    # Leftmost interval: (-inf, midpoint]
                    if j + 1 < len(self.wealth_grid):
                        mid_log = 0.5 * (np.log(self.wealth_grid[j] / Wi_after_cashflow) + 
                                        np.log(self.wealth_grid[j+1] / Wi_after_cashflow))
                        z_mid = (mid_log - mean_log_return) / sigma
                        probs[j] = stats.norm.cdf(z_mid)
                    else:
                        probs[j] = stats.norm.pdf(z_score)
                elif j == len(self.wealth_grid) - 1:
                    # Rightmost interval: [midpoint, +inf)
                    mid_log = 0.5 * (np.log(self.wealth_grid[j-1] / Wi_after_cashflow) + 
                                    np.log(self.wealth_grid[j] / Wi_after_cashflow))
                    z_mid = (mid_log - mean_log_return) / sigma
                    probs[j] = 1.0 - stats.norm.cdf(z_mid)
                else:
                    # Middle intervals: [left_mid, right_mid]
                    left_mid_log = 0.5 * (np.log(self.wealth_grid[j-1] / Wi_after_cashflow) + 
                                         np.log(self.wealth_grid[j] / Wi_after_cashflow))
                    right_mid_log = 0.5 * (np.log(self.wealth_grid[j] / Wi_after_cashflow) + 
                                          np.log(self.wealth_grid[j+1] / Wi_after_cashflow))
                    
                    z_left = (left_mid_log - mean_log_return) / sigma
                    z_right = (right_mid_log - mean_log_return) / sigma
                    
                    probs[j] = stats.norm.cdf(z_right) - stats.norm.cdf(z_left)
        
        # Ensure probabilities are non-negative and finite
        probs = np.maximum(probs, 0.0)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        
        # Normalize probabilities to sum to 1
        total_prob = np.sum(probs)
        if total_prob > 1e-10:  # Avoid division by very small numbers
            probs = probs / total_prob
        else:
            # Fallback: uniform distribution
            probs = np.ones(len(self.wealth_grid)) / len(self.wealth_grid)
        
        return probs
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the GBWM problem using backward dynamic programming
        
        Returns:
            Tuple of (value_function, optimal_policy)
        """
        start_time = time.time()
        
        # Initialize value function V(Wi, t) and policy μ*(Wi, t)
        T = self.config.time_horizon
        n_wealth = len(self.wealth_grid)
        
        V = np.zeros((n_wealth, T + 1))  # Value function
        policy = np.zeros((n_wealth, T), dtype=int)  # Optimal portfolio indices
        
        # Terminal condition: V(Wi, T) = 1 if Wi ≥ G, 0 otherwise
        V[:, T] = (self.wealth_grid >= self.config.goal_wealth).astype(float)
        
        self.logger.info("Starting backward recursion...")
        
        # Backward recursion
        for t in range(T - 1, -1, -1):
            if t % 5 == 0:  # Progress update every 5 time steps
                self.logger.info(f"Processing time period {t}")
            
            for i in range(n_wealth):
                Wi = self.wealth_grid[i]
                
                # Check for bankruptcy (simplified - skip for base case)
                if Wi <= 0:
                    V[i, t] = 0.0
                    policy[i, t] = 0  # Default to most conservative portfolio
                    continue
                
                # Optimize over portfolio choices
                best_value = 0.0
                best_portfolio = 0
                
                for mu_idx, (mu, sigma) in enumerate(zip(self.mu_array, self.sigma_array)):
                    # Compute transition probabilities
                    transition_probs = self._compute_transition_probabilities(Wi, mu, sigma)
                    
                    # Expected value = sum of (probability × next-period value)
                    expected_value = np.sum(transition_probs * V[:, t + 1])
                    
                    if expected_value > best_value:
                        best_value = expected_value
                        best_portfolio = mu_idx
                
                V[i, t] = max(0.0, min(1.0, best_value))  # Ensure bounds [0,1]
                policy[i, t] = best_portfolio
        
        self.solve_time = time.time() - start_time
        
        # Store results
        self.value_function = V
        self.policy = policy
        
        # Get optimal probability for initial wealth
        optimal_prob = V[self.initial_wealth_idx, 0]
        
        self.logger.info(f"✅ DP solved in {self.solve_time:.2f} seconds")
        self.logger.info(f"✅ Optimal success probability: {optimal_prob:.3f}")
        
        return V, policy
    
    def get_optimal_probability(self) -> float:
        """Get the optimal probability of reaching the goal"""
        if self.value_function is None:
            raise ValueError("Must solve the problem first using solve()")
        
        return self.value_function[self.initial_wealth_idx, 0]
    
    def get_optimal_strategy(self, wealth: float, time: int) -> Tuple[int, float, float]:
        """
        Get optimal portfolio choice for given wealth and time
        
        Returns:
            Tuple of (portfolio_index, expected_return, volatility)
        """
        if self.policy is None:
            raise ValueError("Must solve the problem first using solve()")
        
        if time >= self.config.time_horizon:
            return 0, 0.0, 0.0
        
        # Find closest wealth grid point
        wealth_idx = np.argmin(np.abs(self.wealth_grid - wealth))
        portfolio_idx = self.policy[wealth_idx, time]
        
        mu = self.mu_array[portfolio_idx]
        sigma = self.sigma_array[portfolio_idx]
        
        return portfolio_idx, mu, sigma
    
    def simulate_trajectory(self, num_simulations: int = 1000, seed: int = None) -> Dict:
        """
        Simulate trajectories using the optimal policy
        
        Returns:
            Dictionary with simulation results
        """
        if self.value_function is None:
            raise ValueError("Must solve the problem first using solve()")
        
        if seed is not None:
            np.random.seed(seed)
        
        successes = 0
        final_wealths = []
        
        for _ in range(num_simulations):
            wealth = self.config.initial_wealth
            
            for t in range(self.config.time_horizon):
                # Get optimal portfolio choice
                portfolio_idx, mu, sigma = self.get_optimal_strategy(wealth, t)
                
                # Apply cash flow
                wealth += self._get_cash_flow(t)
                
                # Evolve wealth with geometric Brownian motion
                if wealth > 0:
                    drift = mu - 0.5 * sigma**2
                    diffusion = sigma * np.random.normal(0, 1)
                    wealth = wealth * np.exp(drift + diffusion)
                    wealth = max(0.0, wealth)  # Cannot be negative
            
            final_wealths.append(wealth)
            if wealth >= self.config.goal_wealth:
                successes += 1
        
        success_rate = successes / num_simulations
        
        return {
            'success_rate': success_rate,
            'mean_final_wealth': np.mean(final_wealths),
            'std_final_wealth': np.std(final_wealths),
            'min_final_wealth': np.min(final_wealths),
            'max_final_wealth': np.max(final_wealths),
            'final_wealths': final_wealths,
            'num_simulations': num_simulations
        }
    
    def get_policy_summary(self) -> Dict:
        """Get a summary of the optimal policy"""
        if self.policy is None:
            raise ValueError("Must solve the problem first using solve()")
        
        policy_summary = {}
        T = self.config.time_horizon
        
        for t in range(T):
            policy_summary[f'time_{t}'] = {}
            
            for i in range(0, len(self.wealth_grid), max(1, len(self.wealth_grid) // 10)):
                wealth = self.wealth_grid[i]
                portfolio_idx = self.policy[i, t]
                mu = self.mu_array[portfolio_idx]
                sigma = self.sigma_array[portfolio_idx]
                
                risk_level = "Conservative" if sigma < 0.08 else "Moderate" if sigma < 0.15 else "Aggressive"
                
                policy_summary[f'time_{t}'][f'wealth_{wealth:.0f}'] = {
                    'portfolio_index': int(portfolio_idx),
                    'expected_return': float(mu),
                    'volatility': float(sigma),
                    'risk_level': risk_level
                }
        
        return policy_summary
    
    def get_results_summary(self) -> Dict:
        """Get comprehensive results summary"""
        if self.value_function is None:
            raise ValueError("Must solve the problem first using solve()")
        
        # Run simulation to validate
        sim_results = self.simulate_trajectory(num_simulations=10000, seed=42)
        
        return {
            'algorithm': 'Dynamic Programming',
            'config': {
                'initial_wealth': self.config.initial_wealth,
                'goal_wealth': self.config.goal_wealth,
                'time_horizon': self.config.time_horizon,
                'num_portfolios': self.config.num_portfolios,
                'grid_size': len(self.wealth_grid),
                'grid_density': self.config.grid_density
            },
            'theoretical_results': {
                'optimal_probability': float(self.get_optimal_probability()),
                'solve_time': float(self.solve_time)
            },
            'simulation_validation': sim_results,
            'grid_info': {
                'wealth_range': [float(self.wealth_grid.min()), float(self.wealth_grid.max())],
                'grid_size': len(self.wealth_grid),
                'initial_wealth_index': int(self.initial_wealth_idx)
            }
        }


def solve_gbwm_dp(initial_wealth: float = 100000,
                  goal_wealth: float = 200000,
                  time_horizon: int = 10,
                  num_portfolios: int = 15,
                  grid_density: float = 3.0) -> GBWMDynamicProgramming:
    """
    Convenience function to solve GBWM problem with specified parameters

    Args:
        initial_wealth: Starting wealth ($)
        goal_wealth: Target wealth ($)
        time_horizon: Time to goal (years)
        num_portfolios: Number of portfolio choices
        grid_density: Wealth grid density parameter

    Returns:
        Solved GBWMDynamicProgramming instance
    """
    config = DPConfig(
        initial_wealth=initial_wealth,
        goal_wealth=goal_wealth,
        time_horizon=time_horizon,
        num_portfolios=num_portfolios,
        grid_density=grid_density
    )

    dp = GBWMDynamicProgramming(config)
    dp.solve()

    return dp


# =============================================================================
# MULTI-GOAL DYNAMIC PROGRAMMING
# =============================================================================

@dataclass
class MultiGoalDPConfig:
    """
    Configuration for Multi-Goal Dynamic Programming algorithm.

    Based on Das et al. (2022) "Dynamic optimization for multi-goals wealth management"

    Key differences from single-goal DPConfig:
    - Uses goal_years instead of single goal_wealth
    - Includes cost and utility functions for multiple goals
    - Optimizes expected total utility, not success probability
    """
    # Initial parameters
    initial_wealth: float = 120000.0  # W(0) - scaled by num_goals^0.85 * 12 * 10000
    time_horizon: int = 16            # T = 16 years (paper default)

    # Goal schedule - timesteps when goals become available
    goal_years: List[int] = None      # e.g., [4, 8, 12, 16] for annual DP or [47, 95, 143, 191] for monthly DP
    
    # Mapping from goal_years (timesteps) to actual years for cost/utility calculation
    # Only needed when goal_years are in different time units (e.g., monthly timesteps)
    goal_timestep_to_year_mapping: Dict[int, int] = None

    # Goal cost function: C(t) = base_cost * growth_rate^t
    base_goal_cost: float = 100000.0  # $100k base
    goal_cost_growth_rate: float = 1.08  # 8% annual growth

    # Goal utility function: U(t) = base_utility + t
    base_utility: float = 10.0
    utility_time_bonus: float = 1.0   # Linear increase with time

    # Portfolio parameters (efficient frontier bounds)
    mu_min: float = 0.0526           # Minimum expected return
    mu_max: float = 0.0886           # Maximum expected return
    sigma_min: float = 0.0374        # Minimum volatility
    sigma_max: float = 0.1954        # Maximum volatility

    # Efficient frontier coefficients (calculated from market data)
    eff_frontier_a: float = None
    eff_frontier_b: float = None
    eff_frontier_c: float = None

    # Algorithm parameters
    num_portfolios: int = 15         # m = number of portfolio choices
    grid_density: float = 1.5        # Wealth grid density (paper uses ~1.5-3.0 for accuracy)

    # Custom portfolio arrays (if provided, overrides efficient frontier calculation)
    # Use these to ensure DP uses same portfolios as RL for fair comparison
    custom_mu_array: np.ndarray = None   # Custom mean returns array (length = num_portfolios)
    custom_sigma_array: np.ndarray = None  # Custom volatility array (length = num_portfolios)

    # Random seed
    random_seed: int = 42

    # Extended goals support (concurrent & partial goals)
    use_extended_goals: bool = False
    extended_goal_config: ExtendedGoalConfig = None

    def __post_init__(self):
        """Initialize defaults and calculate efficient frontier"""
        if self.goal_years is None:
            self.goal_years = [4, 8, 12, 16]  # Default: 4 goals

        if self.eff_frontier_a is None:
            self._calculate_efficient_frontier_params()

        # If extended goals are enabled but no config provided, create from legacy
        if self.use_extended_goals and self.extended_goal_config is None:
            from config.environment_config import GoalConfig
            legacy_config = GoalConfig(goal_years=self.goal_years)
            self.extended_goal_config = ExtendedGoalConfig.from_legacy(
                legacy_config, self.time_horizon
            )

    def _calculate_efficient_frontier_params(self):
        """Calculate efficient frontier coefficients from paper's market data"""
        # Mean returns from Table 1 (Das et al.)
        mu = np.array([0.0493, 0.0770, 0.0886])  # [US Bonds, Intl Stocks, US Stocks]

        # Covariance matrix from Table 1
        Sigma = np.array([
            [ 0.0017, -0.0017, -0.0021],
            [-0.0017,  0.0396,  0.0309],
            [-0.0021,  0.0309,  0.0392]
        ])

        # Calculate efficient frontier using Markowitz formulas
        ones = np.ones(3)
        Sigma_inv = np.linalg.inv(Sigma)
        k = mu.T @ Sigma_inv @ ones
        l = mu.T @ Sigma_inv @ mu
        p = ones.T @ Sigma_inv @ ones

        denominator = l * p - k**2
        g = (l * Sigma_inv @ ones - k * Sigma_inv @ mu) / denominator
        h = (p * Sigma_inv @ mu - k * Sigma_inv @ ones) / denominator

        # σ = √(aμ² + bμ + c)
        self.eff_frontier_a = h.T @ Sigma @ h
        self.eff_frontier_b = 2 * g.T @ Sigma @ h
        self.eff_frontier_c = g.T @ Sigma @ g

    def get_goal_cost(self, time: int) -> float:
        """Calculate goal cost at given time: C(t) = base_cost * growth_rate^t"""
        return self.base_goal_cost * (self.goal_cost_growth_rate ** time)

    def get_goal_utility(self, time: int) -> float:
        """Calculate goal utility at given time: U(t) = base_utility + t"""
        return self.base_utility + self.utility_time_bonus * time


# =============================================================================
# COST/UTILITY VECTOR GENERATOR (for concurrent & partial goals)
# =============================================================================

class CostUtilityVectorGenerator:
    """
    Generates cost/utility vectors for each time period following the paper's algorithm.

    This implements the algorithm from Das et al. (2022) Section 2.3 for handling:
    - Concurrent goals: Multiple goals at the same time period
    - Partial goals: Multiple fulfillment levels per goal

    Algorithm:
    1. For each time period with goals, generate all combinations of goal options
    2. Compute total cost and utility for each combination
    3. Sort by cost (monotonically increasing)
    4. For same cost, keep only highest utility (remove cost duplicates)
    5. Remove dominated options (where prior column has higher/equal utility)
    6. Final vectors have strictly increasing cost AND utility

    Example (3 concurrent goals at year 5 from paper):
        Goal 1: skip(0,0), full(7,100)
        Goal 2: skip(0,0), partial(9,90), full(20,300)
        Goal 3: skip(0,0), partial1(10,40), partial2(20,250), partial3(30,400), full(40,500)

        Generates 2×3×5 = 30 combinations, then filters to ~14 non-dominated options.
    """

    def __init__(self, goal_config: ExtendedGoalConfig):
        """
        Initialize the generator with an extended goal configuration.

        Args:
            goal_config: ExtendedGoalConfig with all goal definitions
        """
        self.goal_config = goal_config
        self.logger = logging.getLogger(__name__)

        # Storage for pre-computed vectors
        self.cost_vectors = {}      # {t: np.array([c0, c1, ..., ck_max])}
        self.utility_vectors = {}   # {t: np.array([u0, u1, ..., uk_max])}
        self.option_mapping = {}    # {t: {k: [(goal_name, option_label), ...]}}
        self.k_max = {}             # {t: number of options at time t}

        # Pre-compute all vectors
        self._precompute_vectors()

    def _precompute_vectors(self):
        """Pre-compute cost/utility vectors for all time periods."""
        T = self.goal_config.time_horizon

        for t in range(T):
            goals_at_t = self.goal_config.get_goals_at_time(t)

            if not goals_at_t:
                # No goals at this time - single option: skip (cost=0, utility=0)
                self.cost_vectors[t] = np.array([0.0])
                self.utility_vectors[t] = np.array([0.0])
                self.option_mapping[t] = {0: []}
                self.k_max[t] = 1
            else:
                self._generate_vectors_for_time(t, goals_at_t)

        self.logger.info(f"Generated cost/utility vectors for {T} time periods")
        goal_years = self.goal_config.get_all_goal_years()
        for year in goal_years:
            self.logger.info(f"  Year {year}: {self.k_max[year]} options (after filtering)")

    def _generate_vectors_for_time(self, t: int, goals: list):
        """
        Generate cost/utility vectors for a single time period with concurrent goals.

        Args:
            t: Time period
            goals: List of GoalDefinition objects at this time
        """
        from itertools import product

        # Step 1: Generate all combinations (Cartesian product of goal options)
        all_options = [goal.options for goal in goals]
        combinations = list(product(*all_options))

        # Step 2: Compute total cost/utility for each combination
        combo_data = []
        for combo in combinations:
            total_cost = sum(opt.cost for opt in combo)
            total_utility = sum(opt.utility for opt in combo)

            # Record which options were selected
            selections = [
                (goals[i].name, combo[i].label)
                for i in range(len(goals))
            ]

            combo_data.append({
                'cost': total_cost,
                'utility': total_utility,
                'selections': selections
            })

        # Step 3: Sort by cost
        combo_data.sort(key=lambda x: (x['cost'], -x['utility']))

        # Step 4: Remove cost duplicates (keep highest utility for same cost)
        filtered_data = self._remove_cost_duplicates(combo_data)

        # Step 5: Remove dominated options
        final_data = self._remove_dominated(filtered_data)

        # Store results
        self.cost_vectors[t] = np.array([c['cost'] for c in final_data])
        self.utility_vectors[t] = np.array([c['utility'] for c in final_data])
        self.option_mapping[t] = {i: c['selections'] for i, c in enumerate(final_data)}
        self.k_max[t] = len(final_data)

    def _remove_cost_duplicates(self, combo_data: list) -> list:
        """
        Remove entries with duplicate costs, keeping the one with highest utility.

        Args:
            combo_data: List of dicts with 'cost', 'utility', 'selections' keys
                        (must be sorted by cost, then -utility)

        Returns:
            Filtered list with unique costs
        """
        if not combo_data:
            return combo_data

        result = [combo_data[0]]
        for entry in combo_data[1:]:
            if entry['cost'] != result[-1]['cost']:
                result.append(entry)
            # If same cost, we already have higher utility (due to sort order)

        return result

    def _remove_dominated(self, combo_data: list) -> list:
        """
        Remove dominated options where prior column has higher/equal utility.

        An option is dominated if there exists another option with:
        - Lower or equal cost AND higher utility, OR
        - Lower cost AND equal or higher utility

        Since list is sorted by cost, we just need to ensure utility is strictly increasing.

        Args:
            combo_data: List of dicts sorted by cost (unique costs)

        Returns:
            Filtered list with strictly increasing cost AND utility
        """
        if not combo_data:
            return combo_data

        result = [combo_data[0]]

        for entry in combo_data[1:]:
            # Only keep if utility is strictly greater than previous
            if entry['utility'] > result[-1]['utility']:
                result.append(entry)

        return result

    def get_vectors(self, t: int) -> tuple:
        """
        Get cost and utility vectors for a specific time period.

        Args:
            t: Time period

        Returns:
            Tuple of (cost_vector, utility_vector) as numpy arrays
        """
        if t not in self.cost_vectors:
            raise ValueError(f"Time period {t} not in range [0, {self.goal_config.time_horizon})")

        return self.cost_vectors[t], self.utility_vectors[t]

    def get_num_options(self, t: int) -> int:
        """Get number of goal options at time t."""
        return self.k_max.get(t, 1)

    def get_option_details(self, t: int, k: int) -> dict:
        """
        Get detailed information about a specific option.

        Args:
            t: Time period
            k: Option index

        Returns:
            Dict with 'cost', 'utility', 'selections' (list of (goal_name, option_label))
        """
        if t not in self.cost_vectors or k >= self.k_max[t]:
            raise ValueError(f"Invalid (t={t}, k={k})")

        return {
            'cost': self.cost_vectors[t][k],
            'utility': self.utility_vectors[t][k],
            'selections': self.option_mapping[t][k]
        }

    def summarize(self) -> dict:
        """Generate a summary of all vectors for debugging/logging."""
        summary = {}
        for t in range(self.goal_config.time_horizon):
            if self.k_max[t] > 1:  # Only include non-trivial periods
                summary[f'year_{t}'] = {
                    'num_options': self.k_max[t],
                    'cost_range': [float(self.cost_vectors[t].min()),
                                   float(self.cost_vectors[t].max())],
                    'utility_range': [float(self.utility_vectors[t].min()),
                                      float(self.utility_vectors[t].max())],
                    'options': [
                        {
                            'k': k,
                            'cost': float(self.cost_vectors[t][k]),
                            'utility': float(self.utility_vectors[t][k]),
                            'selections': self.option_mapping[t][k]
                        }
                        for k in range(min(5, self.k_max[t]))  # Show first 5 options
                    ]
                }
        return summary


class MultiGoalGBWMDP:
    """
    Multi-Goal Dynamic Programming for Goals-Based Wealth Management

    Implements the algorithm from Das et al. (2022) "Dynamic optimization for
    multi-goals wealth management" (Journal of Banking & Finance).

    Key Algorithm:
    --------------
    Optimizes: max E[Σ utilities from fulfilled goals]

    Bellman Equation:
    V(Wi, t) = max_{k ∈ K(t), l ∈ L} [u_k(t) + Σ_j V(Wj, t+1) × q(Wj | Wi - c_k(t), μ_l)]

    Where:
    - k: goal decision (0=skip, 1=take if available and affordable)
    - l: portfolio choice (0 to num_portfolios-1)
    - u_k(t): utility from goal decision k at time t
    - c_k(t): cost of goal decision k at time t
    - q(·|·): transition probability using geometric Brownian motion

    Terminal Condition: V(Wi, T) = 0 (no bequest utility)

    Key Differences from Single-Goal DP:
    ------------------------------------
    1. Single-goal DP optimizes P[W(T) ≥ G] (success probability)
       Multi-goal DP optimizes E[Σ utilities] (expected total utility)

    2. Single-goal DP only makes portfolio decisions
       Multi-goal DP makes BOTH goal-taking AND portfolio decisions

    3. Single-goal DP: terminal V = 1 if W≥G else 0
       Multi-goal DP: terminal V = 0 (utility comes from goals, not terminal wealth)

    Usage:
    ------
    >>> config = MultiGoalDPConfig(
    ...     initial_wealth=120000,
    ...     goal_years=[4, 8, 12, 16],
    ...     time_horizon=16
    ... )
    >>> dp = MultiGoalGBWMDP(config)
    >>> V, goal_policy, portfolio_policy = dp.solve()
    >>> expected_utility = dp.get_expected_utility()
    """

    def __init__(self, config: MultiGoalDPConfig = None):
        """Initialize the multi-goal DP algorithm"""
        self.config = config or MultiGoalDPConfig()
        self.logger = logging.getLogger(__name__)

        # Set random seed
        np.random.seed(self.config.random_seed)

        # Results storage
        self.value_function = None      # V(Wi, t)
        self.goal_policy = None         # k*(Wi, t): 0=skip, 1=take
        self.portfolio_policy = None    # l*(Wi, t): portfolio index
        self.wealth_grid = None
        self.solve_time = None

        # Setup components
        self._setup_efficient_frontier()
        self._setup_wealth_grid()

        self.logger.info(f"Multi-Goal DP initialized: {len(self.config.goal_years)} goals at years {self.config.goal_years}")

    def _setup_efficient_frontier(self):
        """Setup efficient frontier portfolios"""
        # Check if custom portfolio arrays are provided
        if self.config.custom_mu_array is not None and self.config.custom_sigma_array is not None:
            # Use custom arrays (for fair comparison with RL)
            self.mu_array = np.array(self.config.custom_mu_array)
            self.sigma_array = np.array(self.config.custom_sigma_array)

            if len(self.mu_array) != self.config.num_portfolios:
                self.logger.warning(f"Custom μ array length ({len(self.mu_array)}) != num_portfolios ({self.config.num_portfolios})")
                self.config.num_portfolios = len(self.mu_array)

            self.logger.info(f"Using CUSTOM portfolio arrays: {self.config.num_portfolios} portfolios")
            self.logger.info(f"μ range: [{self.mu_array.min():.4f}, {self.mu_array.max():.4f}]")
            self.logger.info(f"σ range: [{self.sigma_array.min():.4f}, {self.sigma_array.max():.4f}]")
        else:
            # Create equally spaced μ values (original behavior)
            self.mu_array = np.linspace(
                self.config.mu_min,
                self.config.mu_max,
                self.config.num_portfolios
            )

            # Calculate corresponding σ using efficient frontier: σ = √(aμ² + bμ + c)
            mu_squared = self.mu_array ** 2
            variance = (self.config.eff_frontier_a * mu_squared +
                       self.config.eff_frontier_b * self.mu_array +
                       self.config.eff_frontier_c)
            self.sigma_array = np.sqrt(np.maximum(variance, 0.0001))

            self.logger.info(f"Efficient frontier: {self.config.num_portfolios} portfolios")
            self.logger.info(f"μ range: [{self.config.mu_min:.4f}, {self.config.mu_max:.4f}]")
            self.logger.info(f"σ range: [{self.sigma_array.min():.4f}, {self.sigma_array.max():.4f}]")

    def _compute_wealth_bounds(self) -> Tuple[float, float]:
        """Compute extreme wealth bounds for grid"""
        W0 = self.config.initial_wealth
        T = self.config.time_horizon

        # Worst case: minimum return + bad luck (-3σ)
        mu_worst = self.config.mu_min
        sigma_worst = self.config.sigma_max

        # Best case: maximum return + good luck (+3σ)
        mu_best = self.config.mu_max
        sigma_best = self.config.sigma_max

        # Account for potential goal costs (can reduce wealth significantly)
        max_total_cost = sum(
            self.config.get_goal_cost(t) for t in self.config.goal_years
        )

        # Extreme scenarios using GBM
        W_min = max(1.0, W0 * np.exp((mu_worst - 0.5 * sigma_worst**2) * T - 3 * sigma_worst * np.sqrt(T)) - max_total_cost)
        W_max = W0 * np.exp((mu_best - 0.5 * sigma_best**2) * T + 3 * sigma_best * np.sqrt(T))

        return W_min, W_max

    def _setup_wealth_grid(self):
        """Create logarithmically-spaced wealth grid"""
        W_min, W_max = self._compute_wealth_bounds()

        # Logarithmic spacing
        ln_W_min = np.log(W_min)
        ln_W_max = np.log(W_max)

        # Grid size based on density parameter
        grid_span = ln_W_max - ln_W_min
        self.i_max = int(np.ceil(grid_span * self.config.grid_density / self.config.sigma_min))
        self.i_max = max(self.i_max, 50)  # Minimum 50 grid points

        # Create grid
        ln_W_grid = np.linspace(ln_W_min, ln_W_max, self.i_max + 1)
        self.wealth_grid = np.exp(ln_W_grid)

        # Ensure initial wealth is in grid
        closest_idx = np.argmin(np.abs(self.wealth_grid - self.config.initial_wealth))
        self.wealth_grid[closest_idx] = self.config.initial_wealth
        self.initial_wealth_idx = closest_idx

        self.logger.info(f"Wealth grid: {len(self.wealth_grid)} points")
        self.logger.info(f"Range: [${self.wealth_grid.min():,.0f}, ${self.wealth_grid.max():,.0f}]")

    def _precompute_transition_matrices(self):
        """
        Pre-compute transition probability matrices for all portfolios.

        This is a major optimization: instead of computing probabilities on-the-fly,
        we compute P(Wj | Wi, portfolio_l) for all (i, j, l) combinations upfront.

        Creates: self.transition_matrices[portfolio_idx][from_wealth_idx, to_wealth_idx]
        """
        n_wealth = len(self.wealth_grid)
        n_portfolios = self.config.num_portfolios

        # Pre-compute log wealth grid and midpoints for efficiency
        ln_W = np.log(self.wealth_grid)

        # Compute midpoints (used for interval probability calculation)
        # midpoints[j] = (ln_W[j-1] + ln_W[j]) / 2, with special handling at boundaries
        ln_midpoints = np.zeros(n_wealth + 1)
        ln_midpoints[0] = -np.inf  # Left boundary
        ln_midpoints[-1] = np.inf   # Right boundary
        ln_midpoints[1:-1] = 0.5 * (ln_W[:-1] + ln_W[1:])

        self.transition_matrices = []

        for portfolio_idx in range(n_portfolios):
            mu = self.mu_array[portfolio_idx]
            sigma = self.sigma_array[portfolio_idx]
            mean_log_return = mu - 0.5 * sigma**2

            # Create transition matrix for this portfolio
            trans_matrix = np.zeros((n_wealth, n_wealth))

            for i in range(n_wealth):
                Wi = self.wealth_grid[i]
                ln_Wi = ln_W[i]

                if Wi <= 0:
                    trans_matrix[i, 0] = 1.0  # Go to lowest wealth
                    continue

                # Vectorized computation: z-scores for interval boundaries
                # Target log wealth intervals: [ln_midpoints[j], ln_midpoints[j+1]]
                # Required log return: ln(Wj/Wi) = ln_Wj - ln_Wi
                # z_left = (ln_midpoints[j] - ln_Wi - mean_log_return) / sigma
                # z_right = (ln_midpoints[j+1] - ln_Wi - mean_log_return) / sigma

                z_left = (ln_midpoints[:-1] - ln_Wi - mean_log_return) / sigma
                z_right = (ln_midpoints[1:] - ln_Wi - mean_log_return) / sigma

                # Interval probabilities using CDF difference
                probs = stats.norm.cdf(z_right) - stats.norm.cdf(z_left)

                # Normalize and ensure valid probabilities
                probs = np.maximum(probs, 0.0)
                total = np.sum(probs)
                if total > 1e-10:
                    probs = probs / total
                else:
                    probs = np.ones(n_wealth) / n_wealth

                trans_matrix[i, :] = probs

            self.transition_matrices.append(trans_matrix)

        self.logger.info(f"Pre-computed {n_portfolios} transition matrices ({n_wealth}x{n_wealth})")

    def _compute_transition_probabilities(self, Wi: float, mu: float, sigma: float) -> np.ndarray:
        """
        Compute transition probabilities P(Wj at t+1 | Wi at t, portfolio μ)

        Uses geometric Brownian motion:
        W(t+1) = W(t) * exp((μ - σ²/2) + σ*Z) where Z ~ N(0,1)

        Note: For the optimized solve(), we use pre-computed matrices instead.
        This method is kept for flexibility (e.g., after-goal wealth lookups).
        """
        if Wi <= 0:
            probs = np.zeros(len(self.wealth_grid))
            probs[0] = 1.0
            return probs

        n_wealth = len(self.wealth_grid)
        ln_W = np.log(self.wealth_grid)
        ln_Wi = np.log(Wi)
        mean_log_return = mu - 0.5 * sigma**2

        # Compute midpoints
        ln_midpoints = np.zeros(n_wealth + 1)
        ln_midpoints[0] = -np.inf
        ln_midpoints[-1] = np.inf
        ln_midpoints[1:-1] = 0.5 * (ln_W[:-1] + ln_W[1:])

        # Vectorized z-score computation
        z_left = (ln_midpoints[:-1] - ln_Wi - mean_log_return) / sigma
        z_right = (ln_midpoints[1:] - ln_Wi - mean_log_return) / sigma

        # Interval probabilities
        probs = stats.norm.cdf(z_right) - stats.norm.cdf(z_left)
        probs = np.maximum(probs, 0.0)

        # Normalize
        total = np.sum(probs)
        if total > 1e-10:
            probs = probs / total
        else:
            probs = np.ones(n_wealth) / n_wealth

        return probs

    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the multi-goal GBWM problem using backward dynamic programming.

        Algorithm:
        1. Pre-compute transition matrices for all portfolios (optimization)
        2. Initialize V(Wi, T) = 0 for all Wi (no bequest utility)
        3. For t = T-1, T-2, ..., 0:
           For each wealth Wi:
             a. Determine available goal decisions K(t)
             b. For each (goal_decision, portfolio) pair:
                - Compute immediate utility
                - Compute wealth after goal cost
                - Compute expected future value using pre-computed matrices
             c. Select optimal (k*, l*) maximizing total value

        Returns:
            Tuple of (value_function, goal_policy, portfolio_policy)
        """
        # Dispatch to extended solver if enabled
        if self.config.use_extended_goals:
            return self._solve_extended()

        start_time = time.time()

        T = self.config.time_horizon
        n_wealth = len(self.wealth_grid)
        n_portfolios = self.config.num_portfolios

        # Pre-compute transition matrices (major optimization)
        self._precompute_transition_matrices()

        # Initialize arrays
        V = np.zeros((n_wealth, T + 1))
        goal_policy = np.zeros((n_wealth, T), dtype=int)
        portfolio_policy = np.zeros((n_wealth, T), dtype=int)

        # Terminal condition: V(Wi, T) = 0 (no bequest utility)
        V[:, T] = 0.0

        self.logger.info("Starting backward recursion for multi-goal DP...")

        # Pre-compute goal costs and utilities for all years
        # Handle timestep to year mapping for cost/utility calculation
        if self.config.goal_timestep_to_year_mapping:
            # Use mapping to get actual years for cost/utility calculation
            goal_costs = {timestep: self.config.get_goal_cost(year) 
                         for timestep, year in self.config.goal_timestep_to_year_mapping.items()}
            goal_utilities = {timestep: self.config.get_goal_utility(year) 
                             for timestep, year in self.config.goal_timestep_to_year_mapping.items()}
        else:
            # Direct timestep = year (annual DP case)
            goal_costs = {year: self.config.get_goal_cost(year) for year in self.config.goal_years}
            goal_utilities = {year: self.config.get_goal_utility(year) for year in self.config.goal_years}

        # Backward recursion
        for t in range(T - 1, -1, -1):
            if t % 4 == 0:
                self.logger.info(f"Processing time period {t}")

            # Check if this is a goal year (goals are available at time t matching goal_years)
            # This aligns with the RL environment's _is_goal_available() logic
            is_goal_year = t in self.config.goal_years
            effective_year = t  # For cost/utility lookup

            # Handle terminal goal: goal at year T is processed at t=T-1
            # (matching RL env's special handling for time_horizon goals)
            if not is_goal_year and t == T - 1 and T in self.config.goal_years:
                is_goal_year = True
                effective_year = T

            if is_goal_year:
                goal_cost = goal_costs[effective_year]
                goal_utility = goal_utilities[effective_year]
            else:
                goal_cost = 0.0
                goal_utility = 0.0

            # Vectorized computation for skip-goal case (always available)
            # For each portfolio, compute expected future value
            V_next = V[:, t + 1]

            # Case 1: Skip goal (k=0, utility=0, cost=0)
            # expected_future[i, portfolio] = trans_matrix[portfolio][i, :] @ V_next
            skip_values = np.zeros((n_wealth, n_portfolios))
            for portfolio_idx in range(n_portfolios):
                skip_values[:, portfolio_idx] = self.transition_matrices[portfolio_idx] @ V_next

            # Best skip values and portfolios
            best_skip_portfolio = np.argmax(skip_values, axis=1)
            best_skip_value = np.max(skip_values, axis=1)

            # Initialize with skip values
            V[:, t] = best_skip_value
            goal_policy[:, t] = 0
            portfolio_policy[:, t] = best_skip_portfolio

            # Case 2: Take goal (k=1) - only if goal year
            if is_goal_year:
                for i in range(n_wealth):
                    Wi = self.wealth_grid[i]

                    # Check if affordable
                    if Wi < goal_cost:
                        continue

                    # Wealth after taking goal
                    W_after_goal = Wi - goal_cost

                    if W_after_goal <= 0:
                        continue

                    # Find the wealth grid index closest to W_after_goal
                    # (This is an approximation for efficiency)
                    after_goal_idx = np.argmin(np.abs(self.wealth_grid - W_after_goal))

                    # For take-goal case, compute expected future value from after-goal wealth
                    take_values = np.zeros(n_portfolios)
                    for portfolio_idx in range(n_portfolios):
                        take_values[portfolio_idx] = self.transition_matrices[portfolio_idx][after_goal_idx, :] @ V_next

                    # Total value = goal_utility + best expected future
                    best_take_portfolio = np.argmax(take_values)
                    best_take_value = goal_utility + take_values[best_take_portfolio]

                    # Compare with skip
                    if best_take_value > V[i, t]:
                        V[i, t] = best_take_value
                        goal_policy[i, t] = 1
                        portfolio_policy[i, t] = best_take_portfolio

        self.solve_time = time.time() - start_time

        # Store results
        self.value_function = V
        self.goal_policy = goal_policy
        self.portfolio_policy = portfolio_policy

        # Get expected utility from initial state
        expected_utility = V[self.initial_wealth_idx, 0]

        self.logger.info(f"Multi-Goal DP solved in {self.solve_time:.2f} seconds")
        self.logger.info(f"Expected total utility from initial state: {expected_utility:.2f}")

        return V, goal_policy, portfolio_policy

    def _solve_extended(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the extended multi-goal DP with concurrent and partial goals.

        This implements the algorithm from Das et al. (2022) Section 2.3 for handling:
        - Concurrent goals: Multiple goals at the same time period
        - Partial goals: Multiple fulfillment levels per goal

        Key difference from standard solve():
        - Instead of binary goal decisions (skip/take), uses cost/utility vectors
        - goal_policy stores the option index k (not binary 0/1)
        - Option k=0 is always skip (cost=0, utility=0)

        Bellman Equation:
        V(Wi, t) = max_{k ∈ {0,...,k_max}, l ∈ L} [u_k(t) + Σ_j V(Wj, t+1) × q(Wj | Wi - c_k(t), μ_l)]

        Returns:
            Tuple of (value_function, goal_policy, portfolio_policy)
        """
        start_time = time.time()

        T = self.config.time_horizon
        n_wealth = len(self.wealth_grid)
        n_portfolios = self.config.num_portfolios

        # Pre-compute transition matrices
        self._precompute_transition_matrices()

        # Generate cost/utility vectors using the generator
        self.vector_generator = CostUtilityVectorGenerator(self.config.extended_goal_config)

        # Initialize arrays
        V = np.zeros((n_wealth, T + 1))
        goal_policy = np.zeros((n_wealth, T), dtype=int)  # Now stores k index, not binary
        portfolio_policy = np.zeros((n_wealth, T), dtype=int)

        # Terminal condition: V(Wi, T) = 0 (no bequest utility)
        V[:, T] = 0.0

        self.logger.info("Starting backward recursion for extended multi-goal DP...")

        # Backward recursion
        for t in range(T - 1, -1, -1):
            if t % 4 == 0:
                self.logger.info(f"Processing time period {t}")

            # Get cost/utility vectors for this time period
            cost_vec = self.vector_generator.cost_vectors[t]
            utility_vec = self.vector_generator.utility_vectors[t]
            k_max = self.vector_generator.k_max[t]

            V_next = V[:, t + 1]

            # For each wealth state
            for i in range(n_wealth):
                Wi = self.wealth_grid[i]
                best_value = -np.inf
                best_k = 0
                best_portfolio = 0

                # Evaluate all goal decision options
                for k in range(k_max):
                    cost_k = cost_vec[k]
                    utility_k = utility_vec[k]

                    # Check affordability
                    if Wi < cost_k:
                        continue

                    # Wealth after goal decision
                    W_after = Wi - cost_k

                    # Handle zero wealth case (only valid for k=0, skip)
                    if W_after <= 0 and cost_k > 0:
                        continue

                    # Find closest grid point for post-goal wealth
                    if W_after <= 0:
                        after_idx = 0
                    else:
                        after_idx = np.argmin(np.abs(self.wealth_grid - W_after))

                    # Evaluate all portfolios for this goal decision
                    for portfolio_idx in range(n_portfolios):
                        expected_future = self.transition_matrices[portfolio_idx][after_idx, :] @ V_next
                        total_value = utility_k + expected_future

                        if total_value > best_value:
                            best_value = total_value
                            best_k = k
                            best_portfolio = portfolio_idx

                V[i, t] = best_value if best_value > -np.inf else 0.0
                goal_policy[i, t] = best_k
                portfolio_policy[i, t] = best_portfolio

        self.solve_time = time.time() - start_time

        # Store results
        self.value_function = V
        self.goal_policy = goal_policy
        self.portfolio_policy = portfolio_policy

        # Get expected utility from initial state
        expected_utility = V[self.initial_wealth_idx, 0]

        self.logger.info(f"Extended Multi-Goal DP solved in {self.solve_time:.2f} seconds")
        self.logger.info(f"Expected total utility from initial state: {expected_utility:.2f}")

        # Log summary of goal decisions at initial wealth
        self.logger.info("Goal decision summary at initial wealth:")
        for year in self.config.extended_goal_config.get_all_goal_years():
            k = goal_policy[self.initial_wealth_idx, year]
            details = self.vector_generator.get_option_details(year, k)
            self.logger.info(f"  Year {year}: k={k}, cost=${details['cost']:,.0f}, "
                           f"utility={details['utility']:.1f}")

        return V, goal_policy, portfolio_policy

    def get_goal_decisions(self, wealth: float, time: int) -> Dict:
        """
        Get detailed goal decisions for a given state when using extended goals.

        Args:
            wealth: Current wealth
            time: Current time period

        Returns:
            Dict with:
            - 'k': option index
            - 'cost': total cost of selected options
            - 'utility': total utility of selected options
            - 'selections': list of (goal_name, option_label) tuples
        """
        if not self.config.use_extended_goals:
            raise ValueError("Extended goals not enabled. Use get_optimal_strategy() instead.")

        if self.goal_policy is None:
            raise ValueError("Must solve the problem first using solve()")

        if time >= self.config.time_horizon:
            return {'k': 0, 'cost': 0.0, 'utility': 0.0, 'selections': []}

        # Find closest wealth grid point
        wealth_idx = np.argmin(np.abs(self.wealth_grid - wealth))
        k = self.goal_policy[wealth_idx, time]

        details = self.vector_generator.get_option_details(time, k)
        details['k'] = int(k)  # Add the k index to the result
        return details

    def get_expected_utility(self) -> float:
        """Get the expected total utility from the initial wealth state"""
        if self.value_function is None:
            raise ValueError("Must solve the problem first using solve()")
        return self.value_function[self.initial_wealth_idx, 0]

    def get_optimal_strategy(self, wealth: float, time: int) -> Tuple[int, int, float, float]:
        """
        Get optimal strategy for given wealth and time.

        Returns:
            Tuple of (goal_action, portfolio_index, mu, sigma)
            - goal_action: 0=skip, 1=take goal (if available)
            - portfolio_index: optimal portfolio choice
            - mu: expected return of optimal portfolio
            - sigma: volatility of optimal portfolio
        """
        if self.goal_policy is None or self.portfolio_policy is None:
            raise ValueError("Must solve the problem first using solve()")

        if time >= self.config.time_horizon:
            return 0, 0, self.mu_array[0], self.sigma_array[0]

        # Find closest wealth grid point
        wealth_idx = np.argmin(np.abs(self.wealth_grid - wealth))

        goal_action = self.goal_policy[wealth_idx, time]
        portfolio_idx = self.portfolio_policy[wealth_idx, time]

        mu = self.mu_array[portfolio_idx]
        sigma = self.sigma_array[portfolio_idx]

        return int(goal_action), int(portfolio_idx), float(mu), float(sigma)

    def simulate_trajectory(self, num_simulations: int = 1000, seed: int = None) -> Dict:
        """
        Simulate trajectories using the optimal multi-goal policy.

        Returns:
            Dictionary with simulation results including:
            - mean_utility: average total utility achieved
            - goal_achievement_by_year: fraction of simulations taking each goal
            - final_wealth statistics
            - For extended goals: detailed per-goal option selection statistics
        """
        if self.value_function is None:
            raise ValueError("Must solve the problem first using solve()")

        if seed is not None:
            np.random.seed(seed)

        # Dispatch to extended simulation if enabled
        if self.config.use_extended_goals:
            return self._simulate_extended(num_simulations, seed)

        total_utilities = []
        goal_taken_counts = {year: 0 for year in self.config.goal_years}
        final_wealths = []

        for _ in range(num_simulations):
            wealth = self.config.initial_wealth
            total_utility = 0.0

            for t in range(self.config.time_horizon):
                # Get optimal strategy
                goal_action, portfolio_idx, mu, sigma = self.get_optimal_strategy(wealth, t)

                # Check if goal year and handle goal decision
                # Aligns with RL environment's _is_goal_available() logic
                is_goal_year = t in self.config.goal_years
                effective_year = t

                # Handle terminal goal: goal at year T is processed at t=T-1
                if not is_goal_year and t == self.config.time_horizon - 1 and self.config.time_horizon in self.config.goal_years:
                    is_goal_year = True
                    effective_year = self.config.time_horizon

                if is_goal_year:
                    goal_cost = self.config.get_goal_cost(effective_year)
                    goal_utility = self.config.get_goal_utility(effective_year)

                    # Re-evaluate affordability at actual wealth
                    if goal_action == 1 and wealth >= goal_cost:
                        total_utility += goal_utility
                        wealth -= goal_cost
                        goal_taken_counts[effective_year] += 1

                # Wealth evolution with GBM (if not terminal)
                if t < self.config.time_horizon - 1 and wealth > 0:
                    drift = mu - 0.5 * sigma**2
                    diffusion = sigma * np.random.normal(0, 1)
                    wealth = wealth * np.exp(drift + diffusion)
                    wealth = max(0.0, wealth)

            total_utilities.append(total_utility)
            final_wealths.append(wealth)

        # Calculate goal achievement rates
        goal_achievement = {
            year: count / num_simulations
            for year, count in goal_taken_counts.items()
        }

        return {
            'mean_utility': np.mean(total_utilities),
            'std_utility': np.std(total_utilities),
            'min_utility': np.min(total_utilities),
            'max_utility': np.max(total_utilities),
            'goal_achievement_by_year': goal_achievement,
            'total_goals_achieved_mean': sum(goal_achievement.values()),
            'mean_final_wealth': np.mean(final_wealths),
            'std_final_wealth': np.std(final_wealths),
            'num_simulations': num_simulations
        }

    def _simulate_extended(self, num_simulations: int, seed: int = None) -> Dict:
        """
        Simulate trajectories for extended goals with concurrent/partial goals.

        Tracks detailed statistics for each goal including:
        - Which option was selected (skip, partial, full)
        - Per-goal achievement rates
        - Per-option selection frequencies
        """
        if seed is not None:
            np.random.seed(seed)

        total_utilities = []
        final_wealths = []

        # Initialize per-goal tracking
        # goal_outcomes[goal_name] = {option_label: count}
        goal_outcomes = {}
        for goal in self.config.extended_goal_config.goals:
            goal_outcomes[goal.name] = {opt.label: 0 for opt in goal.options}

        # Track k-option selections at each time with goals
        goal_years = self.config.extended_goal_config.get_all_goal_years()
        k_selections = {year: {} for year in goal_years}

        for _ in range(num_simulations):
            wealth = self.config.initial_wealth
            total_utility = 0.0

            for t in range(self.config.time_horizon):
                # Get optimal strategy using standard method (works for extended too)
                _, portfolio_idx, mu, sigma = self.get_optimal_strategy(wealth, t)

                # Get goal decision details for extended goals
                decision = self.get_goal_decisions(wealth, t)
                k = decision['k']
                cost_k = decision['cost']
                utility_k = decision['utility']
                selections = decision['selections']

                # Track k selections at goal years
                if t in goal_years:
                    if k not in k_selections[t]:
                        k_selections[t][k] = 0
                    k_selections[t][k] += 1

                # Apply goal decision if affordable
                if wealth >= cost_k:
                    total_utility += utility_k
                    wealth -= cost_k

                    # Track per-goal option selections
                    for goal_name, option_label in selections:
                        if goal_name in goal_outcomes:
                            if option_label in goal_outcomes[goal_name]:
                                goal_outcomes[goal_name][option_label] += 1

                # Wealth evolution with GBM (if not terminal)
                if t < self.config.time_horizon - 1 and wealth > 0:
                    drift = mu - 0.5 * sigma**2
                    diffusion = sigma * np.random.normal(0, 1)
                    wealth = wealth * np.exp(drift + diffusion)
                    wealth = max(0.0, wealth)

            total_utilities.append(total_utility)
            final_wealths.append(wealth)

        # Convert counts to rates
        goal_option_rates = {}
        for goal_name, options in goal_outcomes.items():
            goal_option_rates[goal_name] = {
                label: count / num_simulations
                for label, count in options.items()
            }

        # Calculate goal achievement (any non-skip option)
        goal_achievement = {}
        for goal_name, rates in goal_option_rates.items():
            skip_rate = rates.get('skip', 0.0)
            goal_achievement[goal_name] = 1.0 - skip_rate

        # Convert k selection counts to rates
        k_selection_rates = {}
        for year, k_counts in k_selections.items():
            k_selection_rates[year] = {
                k: count / num_simulations
                for k, count in k_counts.items()
            }

        return {
            'mean_utility': np.mean(total_utilities),
            'std_utility': np.std(total_utilities),
            'min_utility': np.min(total_utilities),
            'max_utility': np.max(total_utilities),
            'goal_achievement': goal_achievement,
            'goal_option_rates': goal_option_rates,
            'k_selection_rates': k_selection_rates,
            'mean_final_wealth': np.mean(final_wealths),
            'std_final_wealth': np.std(final_wealths),
            'num_simulations': num_simulations
        }

    def get_results_summary(self) -> Dict:
        """Get comprehensive results summary"""
        if self.value_function is None:
            raise ValueError("Must solve the problem first using solve()")

        # Run validation simulation
        sim_results = self.simulate_trajectory(num_simulations=10000, seed=42)

        return {
            'algorithm': 'Multi-Goal Dynamic Programming',
            'config': {
                'initial_wealth': self.config.initial_wealth,
                'goal_years': self.config.goal_years,
                'num_goals': len(self.config.goal_years),
                'time_horizon': self.config.time_horizon,
                'num_portfolios': self.config.num_portfolios,
                'grid_size': len(self.wealth_grid),
                'base_goal_cost': self.config.base_goal_cost,
                'cost_growth_rate': self.config.goal_cost_growth_rate
            },
            'theoretical_results': {
                'expected_utility': float(self.get_expected_utility()),
                'solve_time': float(self.solve_time)
            },
            'simulation_validation': sim_results,
            'grid_info': {
                'wealth_range': [float(self.wealth_grid.min()), float(self.wealth_grid.max())],
                'grid_size': len(self.wealth_grid),
                'initial_wealth_index': int(self.initial_wealth_idx)
            }
        }


def solve_multi_goal_dp(
    num_goals: int = 4,
    initial_wealth: float = None,
    time_horizon: int = 16,
    num_portfolios: int = 15,
    grid_density: float = 1.5
) -> MultiGoalGBWMDP:
    """
    Convenience function to solve multi-goal GBWM problem.

    Args:
        num_goals: Number of goals (1, 2, 4, 8, or 16)
        initial_wealth: Starting wealth (calculated from num_goals if None)
        time_horizon: Investment horizon in years
        num_portfolios: Number of portfolio choices on efficient frontier
        grid_density: Wealth grid density parameter

    Returns:
        Solved MultiGoalGBWMDP instance
    """
    # Set goal years based on num_goals
    if num_goals == 1:
        goal_years = [16]
    elif num_goals == 2:
        goal_years = [8, 16]
    elif num_goals == 4:
        goal_years = [4, 8, 12, 16]
    elif num_goals == 8:
        goal_years = [2, 4, 6, 8, 10, 12, 14, 16]
    elif num_goals == 16:
        goal_years = list(range(1, 17))
    else:
        raise ValueError(f"Unsupported number of goals: {num_goals}")

    # Calculate initial wealth if not provided (paper formula)
    if initial_wealth is None:
        initial_wealth = 12 * (num_goals ** 0.85) * 10000

    config = MultiGoalDPConfig(
        initial_wealth=initial_wealth,
        goal_years=goal_years,
        time_horizon=time_horizon,
        num_portfolios=num_portfolios,
        grid_density=grid_density
    )

    dp = MultiGoalGBWMDP(config)
    dp.solve()

    return dp


def solve_extended_dp(
    extended_goal_config: ExtendedGoalConfig,
    initial_wealth: float = None,
    time_horizon: int = None,
    num_portfolios: int = 15,
    grid_density: float = 1.5
) -> MultiGoalGBWMDP:
    """
    Convenience function to solve extended multi-goal DP with concurrent/partial goals.

    This enables the full paper algorithm from Das et al. (2022) Section 2.3 with:
    - Concurrent goals: Multiple goals at the same time period
    - Partial goals: Multiple fulfillment levels per goal

    Args:
        extended_goal_config: ExtendedGoalConfig with all goal definitions
        initial_wealth: Starting wealth (uses default if None)
        time_horizon: Investment horizon (uses extended_goal_config.time_horizon if None)
        num_portfolios: Number of portfolio choices on efficient frontier
        grid_density: Wealth grid density parameter

    Returns:
        Solved MultiGoalGBWMDP instance with extended goal support

    Example:
        >>> # Create goals with partial options
        >>> car_goal = GoalDefinition(
        ...     name="Car",
        ...     year=8,
        ...     options=[
        ...         GoalOption(cost=0, utility=0, label="skip"),
        ...         GoalOption(cost=28000, utility=80, label="basic"),
        ...         GoalOption(cost=50000, utility=300, label="full"),
        ...     ]
        ... )
        >>> tuition = GoalDefinition(
        ...     name="College",
        ...     year=8,  # Concurrent with car
        ...     options=[
        ...         GoalOption(cost=0, utility=0, label="skip"),
        ...         GoalOption(cost=80000, utility=500, label="full"),
        ...     ]
        ... )
        >>> config = ExtendedGoalConfig(goals=[car_goal, tuition], time_horizon=16)
        >>> dp = solve_extended_dp(config, initial_wealth=200000)
        >>> print(f"Expected utility: {dp.get_expected_utility():.2f}")
    """
    # Use time horizon from extended config if not specified
    if time_horizon is None:
        time_horizon = extended_goal_config.time_horizon

    # Default initial wealth based on number of goals
    if initial_wealth is None:
        num_goals = len(extended_goal_config.goals)
        initial_wealth = 12 * (max(num_goals, 1) ** 0.85) * 10000

    # Extract goal years from extended config for legacy compatibility
    goal_years = extended_goal_config.get_all_goal_years()
    if not goal_years:
        goal_years = [time_horizon]  # Default to terminal goal if no goals specified

    config = MultiGoalDPConfig(
        initial_wealth=initial_wealth,
        goal_years=goal_years,
        time_horizon=time_horizon,
        num_portfolios=num_portfolios,
        grid_density=grid_density,
        use_extended_goals=True,
        extended_goal_config=extended_goal_config
    )

    dp = MultiGoalGBWMDP(config)
    dp.solve()

    return dp


def solve_paper_example_section23(
    initial_wealth: float = 100000,
    time_horizon: int = 11,
    num_portfolios: int = 15,
    grid_density: float = 1.5
) -> MultiGoalGBWMDP:
    """
    Solve the exact example from Das et al. (2022) Section 2.3.

    This creates the three concurrent goals at year 5 from the paper:
    - Goal 1: All-or-nothing (cost 0 or 7, utility 0 or 100)
    - Goal 2: Partial (cost 0, 9, or 20; utility 0, 90, or 300)
    - Goal 3: Partial (cost 0, 10, 20, 30, or 40; utility 0, 40, 250, 400, or 500)

    This is useful for verifying the implementation against the paper.

    Args:
        initial_wealth: Starting wealth
        time_horizon: Investment horizon (default 11 as in paper example)
        num_portfolios: Number of portfolio choices
        grid_density: Wealth grid density

    Returns:
        Solved MultiGoalGBWMDP with paper example configuration
    """
    extended_config = ExtendedGoalConfig.create_paper_example_section23()

    return solve_extended_dp(
        extended_goal_config=extended_config,
        initial_wealth=initial_wealth,
        time_horizon=time_horizon,
        num_portfolios=num_portfolios,
        grid_density=grid_density
    )