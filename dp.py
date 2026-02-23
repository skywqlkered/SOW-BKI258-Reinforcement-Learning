from environment import MouseEnv
import numpy as np

# Shared helper function

def get_action_value(state, action, value_function, gamma):
    # Get reward and next state for this state-action pair
    reward = MouseEnv.get_reward(state, action)
    state_prime = MouseEnv.get_next_state(state, action)

    # If transition is terminal, do not bootstrap from next-state value
    if reward == MouseEnv.lose_punishment or reward == MouseEnv.win_reward:
        action_total = reward

    # Otherwise, apply Bellman expectation update for deterministic transition
    else:
        action_total = reward + (gamma * value_function[state_prime])
    
    return action_total

def policy_index_to_positions(policy: dict, rows: int, cols: int):
    "Change policy dict with state indices as keys to a 4D array indexed by cheese and mouse positions"
    position_policy = np.full((rows, cols, rows, cols), np.nan)
    for state, action in policy.items():
        if MouseEnv.is_terminal_obs(state):
            continue
        mouse_pos, cheese_pos, won = MouseEnv.get_state_from_obs(state)
        m_row, m_col = mouse_pos # type: ignore
        c_row, c_col = cheese_pos # type: ignore
        position_policy[c_row, c_col, m_row, m_col] = action
    return position_policy

# Policy iteration functions

def evaluate_policy(policy, theta, gamma):
    """Evaluate a fixed policy with iterative policy evaluation.

    Repeatedly applies Bellman expectation updates until the maximum value change across states is smaller than `theta`.

    Args:
        policy (dict): Mapping from state index to selected action.
        theta (float): Convergence threshold for value updates.
        gamma (float): Discount factor for future returns.

    Returns:
    - dict: Mapping from state index to estimated value under the given policy.
    - list[float]: History of maximum value changes (deltas) across iterations.
    """
    # Initialize all state values to zero
    Qs: dict[int, float] = {state: 0.0 for state in range(MouseEnv.num_of_states)}
    
    # Track stuff for plotting and reporting
    delta_history = []
    delta = 1000

    # Start iterative policy evaluation loop, until convergence (delta < theta)
    while not delta < theta:

        # Reset per-iteration max update
        delta = 0

        # Go over every state
        for state in range(MouseEnv.num_of_states):

            # Skip terminal states since they have no decision and no future return
            if MouseEnv.is_terminal_obs(state):
                continue

            # Get old value
            q = Qs[state]

            # Get policy action and one-step model outcome
            state_total = 0
            action = policy[state]
            
            action_total = get_action_value(state, action, Qs, gamma)

            # Write updated value and update max delta
            state_total = action_total
            Qs[state] = state_total
            delta = max(delta, abs(q - Qs[state]))

        # Store iteration delta for plotting and diagnostics
        delta_history.append(delta)

    # Return converged value function and convergence history
    return Qs, delta_history


def improve_policy(value_function: dict, policy: dict, gamma: float):
    """Improve a policy greedily with respect to a value function.

    For each non-terminal state, evaluates all actions with one-step lookahead and selects the action with the highest estimated return.

    Args:
        value_function (dict): Mapping from state index to estimated value.
        policy (dict): Current policy mapping state index to action.
        gamma (float): Discount factor for future returns.

    Returns:
    - dict: Improved policy mapping from state index to action.
    - bool: Whether the policy was stable (i.e. unchanged) after improvement.
    """
    policy_stable = True
    improved_policy = {}

    # Iterate over every state in the state space
    for state in range(MouseEnv.num_of_states):
        old_action = policy[state]

        # Terminal states do not need policy improvement
        if MouseEnv.is_terminal_obs(state):
            improved_policy[state] = old_action
            continue

        action_values = {}

        # Go over each action from this state and get the reward and next state
        for action in range(MouseEnv.num_of_actions):
            reward = MouseEnv.get_reward(state, action)
            state_prime = MouseEnv.get_next_state(state, action)

            # If transition is terminal, do not bootstrap from the value function
            if reward == MouseEnv.lose_punishment or reward == MouseEnv.win_reward:
                action_total = reward

            # Otherwise, use one-step Bellman lookahead: r + gamma * V(s')
            else:
                action_total = reward + (gamma * value_function[state_prime]) # p(state | action) = 1, so no probabilities included

            action_values[action] = action_total

        # Select the action with highest estimated return
        best_action = max(action_values, key=lambda x: action_values[x])

        # Mark policy as unstable if this state's action changed
        if best_action != old_action:
            policy_stable = False

        # Save the new best action
        improved_policy[state] = best_action

    # Return improved policy and whether it remained unchanged
    return improved_policy, policy_stable


def policy_iteration(theta: float, gamma: float, max_iterations: int = 1000, track_history: bool = False, state_history: list[int] | None = None):
    """Run policy iteration until convergence or iteration cap.

    Alternates policy evaluation and policy improvement, while tracking diagnostics that can be plotted in the notebook.

    Args:
        theta (float): Convergence threshold used by policy evaluation.
        gamma (float): Discount factor for future returns.
        max_iterations (int, optional): Maximum number of outer policy
            iteration steps. Defaults to 1000.

    Returns:
    - If track_history is False:
        - dict: Final policy mapping from state index to action.
        - dict: Final state-value function mapping from state index to value.
    - If track_history is True:
        - the same values as above, plus
        - list[float]: History of maximum value changes (deltas) during each policy evaluation step.
        - dict[int, list[float]]: History of tracked-state values across policy-iteration steps.
    """
    # Start with a deterministic policy for all states
    policy = {state: 0 for state in range(MouseEnv.num_of_states)}

    # Keep terminal states' actions fixed to a valid default value
    for state in range(MouseEnv.num_of_states):
        if MouseEnv.is_terminal_obs(state):
            policy[state] = 0

    # Track stuff for plotting and reporting
    value_function = {state: 0.0 for state in range(MouseEnv.num_of_states)}
    if track_history:
        if state_history is None:
            state_history = [state for state in range(MouseEnv.num_of_states) if not MouseEnv.is_terminal_obs(state)]
        value_history: dict[int, list[float]] = {state: [] for state in state_history}
        evaluation_delta_history = []

    # Start the capped policy iteration loop
    for _ in range(max_iterations):

        # Evaluate the current policy to obtain its state values
        value_function, delta_history = evaluate_policy(policy=policy, theta=theta, gamma=gamma)

        # Track stuff for plotting and reporting
        if track_history:
            for state in state_history: # type: ignore
                value_history[state].append(value_function[state]) # type: ignore
            evaluation_delta_history.append(delta_history[-1] if len(delta_history) > 0 else 0.0) # type: ignore

        # Improve policy greedily with respect to the evaluated value function
        policy, policy_stable = improve_policy(value_function=value_function, policy=policy, gamma=gamma)

        # Stop when the policy is stable
        if policy_stable:
            break

    # Return final policy, final value function, and histories for plotting
    if track_history:
        return policy, value_function, evaluation_delta_history, value_history # type: ignore

    return policy, value_function

# Value iteration functions

def _create_policy_from_values(value_function: dict[int, float], gamma: float = 1.0) -> dict[int, int]:
    """Create a greedy policy from a value function using one-step lookahead.

    For each non-terminal state, selects the action with the highest estimated return.

    Args:
        value_function (dict[int, float]): Value function mapping state index to estimated value.
        gamma (float, optional): Discount factor for future returns. Defaults to 1.0.

    Returns:
        dict[int, int]: Policy mapping state index to the best action (greedy with respect to the value function).
    """
    policy: dict[int, int] = {}
    
    # Iterate over every state in the state space
    for state in range(MouseEnv.num_of_states):
        best_action: tuple[int, float] = (-1, -np.inf)
        
        # Go over each action and evaluate its one-step return
        for action in range(MouseEnv.num_of_actions):
            action_value = get_action_value(state, action, value_function, gamma)
            
            # Track the action with the highest estimated return
            if action_value > best_action[1]:
                best_action = (action, action_value)
        
        # Assign best action to this state's policy
        policy[state] = best_action[0]

    return policy

def value_iteration(theta: float, gamma: float, track_history: bool = False, state_history: list[int] | None = None):
    """Compute optimal state values with value iteration.

    Repeatedly applies Bellman optimality updates to each state until the maximum value change is smaller than `theta`.

    Args:
        theta (float): Convergence threshold for value updates.
        gamma (float): Discount factor for future returns.
        track_history (bool, optional): If True, also returns convergence and selected-state traces. Defaults to False.
        state_history (list[int] | None, optional): States for which value evolution is tracked. If None and track_history=True, all non-terminal states are tracked.

    Returns:
    - If track_history is False:
        - dict: Final policy mapping from state index to action.
        - dict: Final state-value function mapping from state index to value.
    - If track_history is True:
        - the same values as above, plus
        - list[float]: History of maximum value changes (deltas) during each policy evaluation step.
        - dict[int, list[float]]: History of tracked-state values across policy-iteration steps.
    """
    # Initialize all state values to zero
    states = range(MouseEnv.num_of_states)
    value_function = {state: 0.0 for state in states}

    # Set up tracking if requested
    if track_history:
        if state_history is None:
            state_history = [state for state in states if not MouseEnv.is_terminal_obs(state)]
        delta_history: list[float] = []
        value_history: dict[int, list[float]] = {state: [] for state in state_history}

    # Start Bellman optimality iteration loop until convergence
    delta = float("inf")
    while delta >= theta:
        delta = 0.0
        
        # Go over every state and apply Bellman optimality update
        for state in states:
            # Terminal states have zero value with no decision to make
            if MouseEnv.is_terminal_obs(state):
                value_function[state] = 0.0
                continue

            # Get old value for convergence check
            q = value_function[state]
            
            # Evaluate all actions and select the best one
            action_values = [get_action_value(state, action, value_function, gamma) for action in range(MouseEnv.num_of_actions)]
            value_function[state] = max(action_values)

            # Track maximum value change for convergence criterion
            delta = max(delta, abs(q - value_function[state]))

        # Store diagnostics for plotting and reporting
        if track_history:
            delta_history.append(delta) # type: ignore
            for state in state_history: # type: ignore
                value_history[state].append(value_function[state]) # type: ignore

    # Create a greedy policy from the converged optimal value function
    policy = _create_policy_from_values(value_function, gamma)

    # Return values and optional histories
    if track_history:
        return policy, value_function, delta_history, value_history # type: ignore

    return policy, value_function