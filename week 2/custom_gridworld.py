# Dynamic Programming: Policy Iteration and Value Iteration
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import gymnasium as gym
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# GridWorld Environment Implementation

class CustomGridWorld(gym.Env):
    """
    Custom GridWorld environment with configurable rewards, obstacles, and stochastic transitions.
    
    Parameters:
    -----------
    size : tuple
        (rows, cols) dimensions of the grid
    start : tuple
        Starting position (row, col)
    terminals : list of tuples
        Terminal state positions
    obstacles : list of tuples
        Obstacle positions (impassable)
    rewards : dict
        Custom rewards for specific positions {(row, col): reward}
    stochastic : bool
        If True, actions have probabilistic outcomes
    slip_prob : float
        Probability of slipping (moving perpendicular to intended direction)
    """
    
    def __init__(self, size=(4, 4), start=(0, 0), terminals=None, 
                 obstacles=None, rewards=None, stochastic=False, slip_prob=0.1):
        super().__init__()
        
        self.rows, self.cols = size
        self.nS = self.rows * self.cols
        self.nA = 4  # up, right, down, left
        
        self.start = start
        self.terminals = terminals if terminals else [(self.rows-1, self.cols-1)]
        self.obstacles = obstacles if obstacles else []
        self.custom_rewards = rewards if rewards else {}
        self.stochastic = stochastic
        self.slip_prob = slip_prob
        
        # Convert position tuples to state indices
        self.terminal_states = [self._pos_to_state(pos) for pos in self.terminals]
        self.obstacle_states = [self._pos_to_state(pos) for pos in self.obstacles]
        
        # Action names for visualization
        self.action_names = ['↑', '→', '↓', '←']
        self.action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Build transition dynamics
        self.P = self._build_transitions()
    
    def _pos_to_state(self, pos):
        """Convert (row, col) to state index."""
        return pos[0] * self.cols + pos[1]
    
    def _state_to_pos(self, state):
        """Convert state index to (row, col)."""
        return (state // self.cols, state % self.cols)
    
    def _is_valid_pos(self, pos):
        """Check if position is within bounds and not an obstacle."""
        row, col = pos
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        if pos in self.obstacles:
            return False
        return True
    
    def _get_next_pos(self, pos, action):
        """Get next position given current position and action."""
        row, col = pos
        dr, dc = self.action_deltas[action]
        next_pos = (row + dr, col + dc)
        
        # If next position is invalid, stay in current position
        if not self._is_valid_pos(next_pos):
            return pos
        return next_pos
    
    def _get_reward(self, pos):
        """Get reward for a position."""
        if pos in self.custom_rewards:
            return self.custom_rewards[pos]
        if pos in self.terminals:
            return 0.0
        return -1.0  # Default step cost
    
    def _build_transitions(self):
        """Build P(s',r|s,a) for all s,a."""
        P = {}
        
        for s in range(self.nS):
            pos = self._state_to_pos(s)
            P[s] = {a: [] for a in range(self.nA)}
            
            # Terminal states
            if s in self.terminal_states:
                for a in range(self.nA):
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue
            
            # Obstacle states (shouldn't reach here, but handle anyway)
            if s in self.obstacle_states:
                for a in range(self.nA):
                    P[s][a] = [(1.0, s, -10.0, False)]
                continue
            
            # Regular states
            for a in range(self.nA):
                if not self.stochastic:
                    # Deterministic transitions
                    next_pos = self._get_next_pos(pos, a)
                    next_s = self._pos_to_state(next_pos)
                    reward = self._get_reward(next_pos)
                    done = next_s in self.terminal_states
                    P[s][a] = [(1.0, next_s, reward, done)]
                else:
                    # Stochastic transitions
                    # Intended action: (1 - 2*slip_prob) probability
                    # Perpendicular actions: slip_prob each
                    transitions = []
                    
                    # Intended direction
                    intended_prob = 1.0 - 2 * self.slip_prob
                    next_pos = self._get_next_pos(pos, a)
                    next_s = self._pos_to_state(next_pos)
                    reward = self._get_reward(next_pos)
                    done = next_s in self.terminal_states
                    transitions.append((intended_prob, next_s, reward, done))
                    
                    # Perpendicular directions
                    perp_actions = [(a - 1) % 4, (a + 1) % 4]
                    for perp_a in perp_actions:
                        next_pos = self._get_next_pos(pos, perp_a)
                        next_s = self._pos_to_state(next_pos)
                        reward = self._get_reward(next_pos)
                        done = next_s in self.terminal_states
                        transitions.append((self.slip_prob, next_s, reward, done))
                    
                    P[s][a] = transitions
        
        return P
    
    def reset(self, seed=None):
        """Reset environment to starting state."""
        super().reset(seed=seed)
        self.current_state = self._pos_to_state(self.start)
        return self.current_state, {}
    
    def step(self, action):
        """Execute action and return next state, reward, done, info."""
        transitions = self.P[self.current_state][action]
        probs = [t[0] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        _, next_s, reward, done = transitions[idx]
        
        self.current_state = next_s
        return next_s, reward, done, False, {}

# Policy Evaluation - Synchronous Version

def policy_evaluation_sync(env, policy, gamma=0.9, theta=1e-6, max_iterations=1000):
    """
    Synchronous policy evaluation using two arrays.
    
    Bellman equation for V^π:
    V^π(s) = Σ_a π(a|s) * Σ_{s',r} P(s',r|s,a) * [r + γ*V^π(s')]
    
    Parameters:
    -----------
    env : GridWorld environment
    policy : ndarray of shape (nS, nA)
        Policy π(a|s)
    gamma : float
        Discount factor
    theta : float
        Convergence threshold
    max_iterations : int
        Maximum number of iterations
    
    Returns:
    --------
    V : ndarray
        State-value function
    iterations : int
        Number of iterations to converge
    deltas : list
        Maximum change in value per iteration
    """
    V = np.zeros(env.nS)
    V_new = np.zeros(env.nS)
    iterations = 0
    deltas = []
    
    for iteration in range(max_iterations):
        delta = 0
        
        # Sweep through all states
        for s in range(env.nS):
            v = V[s]
            
            # Bellman update
            new_value = 0
            for a in range(env.nA):
                action_prob = policy[s, a]
                for prob, next_s, reward, done in env.P[s][a]:
                    next_value = 0 if done else V[next_s]  # Use old values
                    new_value += action_prob * prob * (reward + gamma * next_value)
            
            V_new[s] = new_value
            delta = max(delta, abs(v - V_new[s]))
        
        # Copy new values to V
        V = V_new.copy()
        deltas.append(delta)
        iterations += 1
        
        # Check convergence
        if delta < theta:
            break
    
    return V, iterations, deltas

# Policy Evaluation - In-Place Version

def policy_evaluation_inplace(env, policy, gamma=0.9, theta=1e-6, max_iterations=1000):
    """
    In-place policy evaluation using single array.
    Updates use immediately available new values.
    
    Typically converges faster than synchronous version.
    
    Parameters:
    -----------
    env : GridWorld environment
    policy : ndarray of shape (nS, nA)
        Policy π(a|s)
    gamma : float
        Discount factor
    theta : float
        Convergence threshold
    max_iterations : int
        Maximum number of iterations
    
    Returns:
    --------
    V : ndarray
        State-value function
    iterations : int
        Number of iterations to converge
    deltas : list
        Maximum change in value per iteration
    """
    V = np.zeros(env.nS)
    iterations = 0
    deltas = []
    
    for iteration in range(max_iterations):
        delta = 0
        
        # Sweep through all states
        for s in range(env.nS):
            v = V[s]
            
            # Bellman update (uses updated values immediately)
            new_value = 0
            for a in range(env.nA):
                action_prob = policy[s, a]
                for prob, next_s, reward, done in env.P[s][a]:
                    next_value = 0 if done else V[next_s]  # May use new values
                    new_value += action_prob * prob * (reward + gamma * next_value)
            
            V[s] = new_value
            delta = max(delta, abs(v - V[s]))
        
        deltas.append(delta)
        iterations += 1
        
        # Check convergence
        if delta < theta:
            break
    
    return V, iterations, deltas

# Policy Improvement

def policy_improvement(env, V, gamma=0.9):
    """
    Greedy policy improvement.
    
    Policy Improvement Theorem:
    π'(s) = argmax_a Q^π(s,a)
         = argmax_a Σ_{s',r} P(s',r|s,a) * [r + γ*V^π(s')]
    
    Parameters:
    -----------
    env : GridWorld environment
    V : ndarray
        Current value function
    gamma : float
        Discount factor
    
    Returns:
    --------
    policy : ndarray of shape (nS, nA)
        Improved policy
    policy_stable : bool
        True if policy didn't change
    """
    policy = np.zeros((env.nS, env.nA))
    policy_stable = True
    
    for s in range(env.nS):
        # Compute Q(s,a) for all actions
        q_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_s, reward, done in env.P[s][a]:
                next_value = 0 if done else V[next_s]
                q_values[a] += prob * (reward + gamma * next_value)
        
        # Greedy action selection
        best_action = np.argmax(q_values)
        
        # Deterministic greedy policy
        policy[s, best_action] = 1.0
    
    return policy

# Policy Iteration - Synchronous

def policy_iteration_sync(env, gamma=0.9, theta=1e-6, max_iterations=100):
    """
    Policy Iteration algorithm (synchronous evaluation).
    
    Algorithm:
    1. Initialize π arbitrarily
    2. Policy Evaluation: Compute V^π
    3. Policy Improvement: π ← greedy(V)
    4. Repeat until convergence
    
    Parameters:
    -----------
    env : GridWorld environment
    gamma : float
        Discount factor
    theta : float
        Convergence threshold for evaluation
    max_iterations : int
        Maximum number of policy iterations
    
    Returns:
    --------
    policy : ndarray
        Optimal policy
    V : ndarray
        Optimal value function
    history : dict
        Training history (iterations, values, policies, times)
    """
    # Initialize with uniform random policy
    policy = np.ones((env.nS, env.nA)) / env.nA
    
    history = {
        'iterations': 0,
        'V_history': [],
        'policy_history': [],
        'eval_iterations': [],
        'times': []
    }
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Policy Evaluation
        V, eval_iters, _ = policy_evaluation_sync(env, policy, gamma, theta)
        
        # Store history
        history['V_history'].append(V.copy())
        history['policy_history'].append(policy.copy())
        history['eval_iterations'].append(eval_iters)
        history['times'].append(time.time() - start_time)
        
        # Policy Improvement
        new_policy = policy_improvement(env, V, gamma)
        
        # Check if policy is stable
        if np.allclose(policy, new_policy):
            history['iterations'] = iteration + 1
            break
        
        policy = new_policy
    
    return policy, V, history

# Policy Iteration - In-Place

def policy_iteration_inplace(env, gamma=0.9, theta=1e-6, max_iterations=100):
    """
    Policy Iteration algorithm (in-place evaluation).
    
    Same as synchronous but uses in-place policy evaluation.
    Typically converges faster.
    
    Parameters:
    -----------
    env : GridWorld environment
    gamma : float
        Discount factor
    theta : float
        Convergence threshold for evaluation
    max_iterations : int
        Maximum number of policy iterations
    
    Returns:
    --------
    policy : ndarray
        Optimal policy
    V : ndarray
        Optimal value function
    history : dict
        Training history
    """
    # Initialize with uniform random policy
    policy = np.ones((env.nS, env.nA)) / env.nA
    
    history = {
        'iterations': 0,
        'V_history': [],
        'policy_history': [],
        'eval_iterations': [],
        'times': []
    }
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Policy Evaluation (in-place)
        V, eval_iters, _ = policy_evaluation_inplace(env, policy, gamma, theta)
        
        # Store history
        history['V_history'].append(V.copy())
        history['policy_history'].append(policy.copy())
        history['eval_iterations'].append(eval_iters)
        history['times'].append(time.time() - start_time)
        
        # Policy Improvement
        new_policy = policy_improvement(env, V, gamma)
        
        # Check if policy is stable
        if np.allclose(policy, new_policy):
            history['iterations'] = iteration + 1
            break
        
        policy = new_policy
    
    return policy, V, history

# Value Iteration - Synchronous

def value_iteration_sync(env, gamma=0.9, theta=1e-6, max_iterations=1000):
    """
    Value Iteration algorithm (synchronous).
    
    Bellman Optimality Equation:
    V*(s) = max_a Σ_{s',r} P(s',r|s,a) * [r + γ*V*(s')]
    
    Combines evaluation and improvement in single update.
    
    Parameters:
    -----------
    env : GridWorld environment
    gamma : float
        Discount factor
    theta : float
        Convergence threshold
    max_iterations : int
        Maximum number of iterations
    
    Returns:
    --------
    policy : ndarray
        Optimal policy
    V : ndarray
        Optimal value function
    history : dict
        Training history
    """
    V = np.zeros(env.nS)
    V_new = np.zeros(env.nS)
    
    history = {
        'iterations': 0,
        'V_history': [],
        'deltas': [],
        'times': []
    }
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        delta = 0
        
        # Sweep through all states
        for s in range(env.nS):
            v = V[s]
            
            # Bellman optimality update
            q_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_s, reward, done in env.P[s][a]:
                    next_value = 0 if done else V[next_s]  # Use old values
                    q_values[a] += prob * (reward + gamma * next_value)
            
            V_new[s] = np.max(q_values)
            delta = max(delta, abs(v - V_new[s]))
        
        # Copy new values
        V = V_new.copy()
        
        # Store history every 10 iterations
        if iteration % 10 == 0 or delta < theta:
            history['V_history'].append(V.copy())
            history['times'].append(time.time() - start_time)
        history['deltas'].append(delta)
        
        # Check convergence
        if delta < theta:
            history['iterations'] = iteration + 1
            break
    
    # Extract greedy policy
    policy = policy_improvement(env, V, gamma)
    
    return policy, V, history

# Value Iteration - In-Place

def value_iteration_inplace(env, gamma=0.9, theta=1e-6, max_iterations=1000):
    """
    Value Iteration algorithm (in-place).
    
    Uses single array for values, updating immediately.
    Typically converges faster than synchronous version.
    
    Parameters:
    -----------
    env : GridWorld environment
    gamma : float
        Discount factor
    theta : float
        Convergence threshold
    max_iterations : int
        Maximum number of iterations
    
    Returns:
    --------
    policy : ndarray
        Optimal policy
    V : ndarray
        Optimal value function
    history : dict
        Training history
    """
    V = np.zeros(env.nS)
    
    history = {
        'iterations': 0,
        'V_history': [],
        'deltas': [],
        'times': []
    }
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        delta = 0
        
        # Sweep through all states
        for s in range(env.nS):
            v = V[s]
            
            # Bellman optimality update (in-place)
            q_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_s, reward, done in env.P[s][a]:
                    next_value = 0 if done else V[next_s]  # May use new values
                    q_values[a] += prob * (reward + gamma * next_value)
            
            V[s] = np.max(q_values)
            delta = max(delta, abs(v - V[s]))
        
        # Store history every 10 iterations
        if iteration % 10 == 0 or delta < theta:
            history['V_history'].append(V.copy())
            history['times'].append(time.time() - start_time)
        history['deltas'].append(delta)
        
        # Check convergence
        if delta < theta:
            history['iterations'] = iteration + 1
            break
    
    # Extract greedy policy
    policy = policy_improvement(env, V, gamma)
    
    return policy, V, history

# Visualization Functions

def plot_value_function(env, V, title="Value Function", ax=None):
    """Plot value function as heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Reshape to grid
    V_grid = V.reshape(env.rows, env.cols)
    
    # Create heatmap
    im = ax.imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', rotation=270, labelpad=20)
    
    # Add value text
    for i in range(env.rows):
        for j in range(env.cols):
            state = i * env.cols + j
            if state in env.terminal_states:
                text = 'T'
            elif state in env.obstacle_states:
                text = 'X'
            else:
                text = f'{V_grid[i, j]:.1f}'
            ax.text(j, i, text, ha='center', va='center', 
                   color='black', fontsize=10, fontweight='bold')
    
    # Grid lines
    ax.set_xticks(np.arange(env.cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(env.rows) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=2)
    ax.tick_params(which='both', size=0, labelbottom=False, labelleft=False)
    
    return ax

def plot_policy(env, policy, title="Policy", ax=None):
    """Plot policy as arrows using quiver plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create grid for background
    grid = np.zeros((env.rows, env.cols))
    for i in range(env.rows):
        for j in range(env.cols):
            state = i * env.cols + j
            if state in env.terminal_states:
                grid[i, j] = 1
            elif state in env.obstacle_states:
                grid[i, j] = -1
    
    ax.imshow(grid, cmap='gray', alpha=0.3, vmin=-1, vmax=1)
    
    # Plot arrows
    for i in range(env.rows):
        for j in range(env.cols):
            state = i * env.cols + j
            
            if state in env.terminal_states or state in env.obstacle_states:
                continue
            
            # Get action probabilities
            action_probs = policy[state]
            best_action = np.argmax(action_probs)
            
            # Arrow direction
            dr, dc = env.action_deltas[best_action]
            
            # Draw arrow
            ax.arrow(j, i, dc * 0.3, dr * 0.3, 
                    head_width=0.2, head_length=0.15,
                    fc='blue', ec='blue', linewidth=2)
    
    # Mark terminals and obstacles
    for i in range(env.rows):
        for j in range(env.cols):
            state = i * env.cols + j
            if state in env.terminal_states:
                ax.text(j, i, 'T', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='green')
            elif state in env.obstacle_states:
                ax.text(j, i, 'X', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='red')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_convergence(histories, labels, title="Convergence Comparison"):
    """Plot convergence curves for multiple algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot iterations
    ax = axes[0]
    for hist, label in zip(histories, labels):
        iterations = range(1, len(hist['V_history']) + 1)
        # Compute max value change between iterations
        changes = []
        for i in range(1, len(hist['V_history'])):
            change = np.max(np.abs(hist['V_history'][i] - hist['V_history'][i-1]))
            changes.append(change)
        if changes:
            ax.semilogy(iterations[1:], changes, marker='o', label=label, linewidth=2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Max Value Change (log scale)', fontsize=12)
    ax.set_title('Convergence Speed', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot wall-clock time
    ax = axes[1]
    for hist, label in zip(histories, labels):
        if 'times' in hist and len(hist['times']) > 0:
            ax.plot(hist['times'], range(1, len(hist['times']) + 1), 
                   marker='o', label=label, linewidth=2)
    
    ax.set_xlabel('Wall-Clock Time (seconds)', fontsize=12)
    ax.set_ylabel('Iteration', fontsize=12)
    ax.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_iterations(env, V_history, policy_history, algorithm_name):
    """Create animation-like visualization of value and policy evolution."""
    n_iters = len(V_history)
    n_plots = min(4, n_iters)  # Show at most 4 iterations
    indices = np.linspace(0, n_iters - 1, n_plots, dtype=int)
    
    fig, axes = plt.subplots(2, n_plots, figsize=(5*n_plots, 10))
    if n_plots == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, i in enumerate(indices):
        # Plot value function
        plot_value_function(env, V_history[i], 
                          f"Iteration {i}", ax=axes[0, idx])
        
        # Plot policy
        if policy_history:
            plot_policy(env, policy_history[i], 
                       f"Iteration {i}", ax=axes[1, idx])
    
    fig.suptitle(f'{algorithm_name} - Evolution Over Time', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
