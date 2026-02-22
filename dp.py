from environment import MouseEnv
import numpy as np
import random


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


def evaluate_policy(policy, env: MouseEnv, theta, gamma):
    """Evaluate a fixed policy with iterative policy evaluation.

    Repeatedly applies Bellman expectation updates until the maximum value change across states is smaller than `theta`.

    Args:
        policy (dict): Mapping from state index to selected action.
        env (MouseEnv): Environment model exposing transition and reward helpers.
        theta (float): Convergence threshold for value updates.
        gamma (float): Discount factor for future returns.

    Returns:
    - dict: Mapping from state index to estimated value under the given policy.
    - list[float]: History of maximum value changes (deltas) across iterations.
    """
    # Initialize all state values to zero
    Vs: dict[float] = {state: 0.0 for state in range(env.num_of_states)} # type:ignore
    
    # Track stuff for plotting and reporting
    delta_history = []
    delta = 1000

    # Start iterative policy evaluation loop, until convergence (delta < theta)
    while not delta < theta:

        # Reset per-iteration max update
        delta = 0

        # Go over every state
        for state in range(env.num_of_states):

            # Skip terminal states since they have no decision and no future return
            if env.is_terminal_obs(state):
                continue

            # Get old value
            v = Vs[state]

            # Get policy action and one-step model outcome
            state_total = 0
            action = policy[state]
            reward = env.get_reward(state, action)
            state_prime = env.get_next_state(state, action)

            # If transition is terminal, do not bootstrap from next-state value
            if reward == env.lose_punishment or reward == env.win_reward:
                action_total = reward

            # Otherwise, apply Bellman expectation update for deterministic transition
            else:
                action_total = reward + (gamma * Vs[state_prime]) # p(state | action) = 1, so no probabilities included

            # Write updated value and update max delta
            state_total = action_total
            Vs[state] = state_total
            delta = max(delta, abs(v - Vs[state]))

        # Store iteration delta for plotting and diagnostics
        delta_history.append(delta)

    # Return converged value function and convergence history
    return Vs, delta_history


def improve_policy(value_function: dict, policy: dict, env: MouseEnv, gamma: float):
    """Improve a policy greedily with respect to a value function.

    For each non-terminal state, evaluates all actions with one-step lookahead and selects the action with the highest estimated return.

    Args:
        value_function (dict): Mapping from state index to estimated value.
        policy (dict): Current policy mapping state index to action.
        env (MouseEnv): Environment model exposing transition and reward helpers.
        gamma (float): Discount factor for future returns.

    Returns:
    - dict: Improved policy mapping from state index to action.
    - bool: Whether the policy was stable (i.e. unchanged) after improvement.
    """
    policy_stable = True
    improved_policy = {}

    # Iterate over every state in the state space
    for state in range(env.num_of_states):
        old_action = policy[state]

        # Terminal states do not need policy improvement
        if env.is_terminal_obs(state):
            improved_policy[state] = old_action
            continue

        action_values = {}

        # Go over each action from this state and get the reward and next state
        for action in range(env.num_of_actions):
            reward = env.get_reward(state, action)
            state_prime = env.get_next_state(state, action)

            # If transition is terminal, do not bootstrap from the value function
            if reward == env.lose_punishment or reward == env.win_reward:
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


def policy_iteration(env: MouseEnv, theta: float, gamma: float, max_iterations: int = 1000):
    """Run policy iteration until convergence or iteration cap.

    Alternates policy evaluation and policy improvement, while tracking diagnostics that can be plotted in the notebook.

    Args:
        env (MouseEnv): Environment model exposing transition and reward helpers.
        theta (float): Convergence threshold used by policy evaluation.
        gamma (float): Discount factor for future returns.
        max_iterations (int, optional): Maximum number of outer policy
            iteration steps. Defaults to 1000.

    Returns:
    - dict: Final policy mapping from state index to action.
    - dict: Final state-value function mapping from state index to value.
    - list[float]: History of maximum value changes (deltas) during each policy evaluation step.
    - bool: Whether the final policy is stable (True) or iteration stopped due to reaching the iteration cap (False).
    """
    # Start with a random policy for all states
    policy = {state: random.randint(0, env.num_of_actions - 1) for state in range(env.num_of_states)}

    # Keep terminal states' actions fixed to a valid default value
    for state in range(env.num_of_states):
        if env.is_terminal_obs(state):
            policy[state] = 0

    # Track stuff for plotting and reporting
    value_function = {state: 0.0 for state in range(env.num_of_states)}
    stable_policy = False
    evaluation_delta_history = []

    # Start the capped policy iteration loop
    for _ in range(max_iterations):

        # Evaluate the current policy to obtain its state values
        value_function, delta_history = evaluate_policy(policy=policy, env=env, theta=theta, gamma=gamma)
        evaluation_delta_history.append(delta_history[-1] if len(delta_history) > 0 else 0.0)

        # Improve policy greedily with respect to the evaluated value function
        policy, policy_stable = improve_policy(value_function=value_function, policy=policy, env=env, gamma=gamma)

        # Stop when the policy is stable
        if policy_stable:
            stable_policy = True
            break

    # Return final policy, final value function, and histories for plotting
    return policy, value_function, evaluation_delta_history, stable_policy