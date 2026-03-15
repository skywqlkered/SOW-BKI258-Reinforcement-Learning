from dp import get_action_value
from environment import MouseEnv
import random


def montecarlo_prediction(policy: dict[int, int], num_of_episodes: int, gamma: float):
    """
    Estimate state values by averaging returns from sampled episodes under the given policy.

    For each episode:
    - Generate an episode using the policy from a random starting state
    - For each state in the episode, calculate the return (discounted cumulative reward)
    - Track the return for each state-visitation
    - After all episodes, estimate each state's value as the average of all returns for that state

    Args:
        policy (dict[int, int]): Mapping from state index to selected action.
        num_of_episodes (int): Number of episodes to generate for estimation.
        gamma (float): Discount factor for future returns.

    Returns:
        dict[int, float]: Mapping from state index to estimated value (average return from policy).
    """
    # Initialize a dictionary to store all returns (G) for each state across episodes
    returns: dict[int, list[float]] = {
        state: [] for state in range(MouseEnv.num_of_states)
    }

    # Generate and process multiple episodes
    for _ in range(num_of_episodes):
        # Generate a single episode following the current policy
        episode = generate_episode(policy)
        # Initialize the return accumulator
        g = 0

        # Track visited states for first-visit Monte Carlo
        visited_states = [state[0] for state in episode]

        # Work backward through the episode to compute returns (backward iteration for discount efficiency)
        while episode:
            # Pop the last (state, action, reward) tuple from the episode
            state, action, reward = episode.pop()
            # Update return using discounted reward: G = R + γ*G
            g = gamma * g + reward
            # For first-visit only: store the return if this is the first visit to the state in this episode
            if state not in visited_states[: len(episode)]:
                returns[state].append(g)

    # Compute state values as the average of collected returns
    values: dict[int, float] = {}
    for state in range(MouseEnv.num_of_states):
        if returns[state]:
            # Average the returns collected for this state
            values[state] = sum(returns[state]) / len(returns[state])
        else:
            # Unvisited states have zero value estimate
            values[state] = 0.0
    return values


def generate_episode(policy, max_steps=100) -> list[tuple[int, int, float]]:
    """
    Generate a single episode by executing the given policy from a random starting state.

    Follows the policy until reaching a terminal state or maximum steps is exceeded.
    Records (state, action, reward) transitions during the trajectory.

    Args:
        policy (dict[int, int]): Mapping from state index to action selected by the policy.
        max_steps (int, optional): Maximum number of steps before forcibly terminating the episode. Defaults to 100.

    Returns:
        list[tuple[int, int, float]]: Sequence of (state, action, reward) tuples representing the episode trajectory.
    """
    # Start from a random state in the environment
    state = random.randrange(0, MouseEnv.num_of_states)
    steps = 0
    episode = []

    # Execute the policy until reaching a terminal state or max_steps
    while not MouseEnv.is_terminal_obs(state) and steps < max_steps:
        # Get the action prescribed by the policy for the current state
        action = policy[state]
        # Query the environment for the immediate reward for this transition
        reward = MouseEnv.get_reward(state, action)
        # Record the transition
        episode.append((state, action, reward))

        # Transition to the next state according to the environment dynamics
        next_state = MouseEnv.get_next_state(state, action)
        state = next_state

        steps += 1

    return episode


def control(values: dict[int, float], discount: float):
    """Improve the policy greedily with respect to estimated state values.

    For each state, selects the action with the highest estimated return using one-step lookahead.

    Args:
        values (dict[int, float]): Mapping from state index to estimated value.
        discount (float): Discount factor for future returns.

    Returns:
        dict[int, int]: Improved policy mapping from state index to greedy action.
    """

    # Initialize the greedy policy
    policy: dict[int, int] = {}

    # Iterate over all states and select the best action for each
    for state in range(MouseEnv.num_of_states):
        best_action = None
        best_value = float("-inf")

        # Evaluate all possible actions from this state using one-step lookahead
        for action in range(MouseEnv.num_of_actions):
            # Get the estimated return for taking this action: r + γ*V(s')
            value = get_action_value(state, action, values, discount)

            # Track the action with the highest estimated return
            if value > best_value:
                best_value = value
                best_action = action

        # Assign the best action to this state's policy
        policy[state] = best_action

    return policy


def montecarlo(num_of_episodes, discount):
    """
    Run Monte Carlo control algorithm to find an optimal policy and state-value function.

    Alternates between policy evaluation (via Monte Carlo prediction) and policy improvement
    until the policy converges (remains unchanged across an iteration).

    Args:
        num_of_episodes (int): Number of episodes to generate during each policy evaluation step.
        discount (float): Discount factor for future returns.

    Returns:
        tuple:
            - dict[int, int]: Converged policy mapping from state index to action.
            - dict[int, float]: Final state-value function mapping from state index to estimated value.
    """
    # Initialize policy randomly for all states
    policy = {
        state: random.randint(0, MouseEnv.num_of_actions - 1)
        for state in range(MouseEnv.num_of_states)
    }

    # Alternate between policy evaluation and policy improvement until convergence
    while True:
        # Policy Evaluation: Estimate state values under the current policy using Monte Carlo
        values = montecarlo_prediction(policy, num_of_episodes, discount)
        # Policy Improvement: Select greedy actions with respect to the estimated values
        new_policy = control(values, discount)
        # Convergence check: stop if the policy is stable (unchanged)
        if new_policy == policy:
            break
        # Update to the improved policy and continue
        policy = new_policy.copy()

    return policy, values

def track_montecarlo_prediction(policy: dict[int, int],
                                num_of_episodes: int = 1000,
                                gamma: float = 0.5,
                                convergence_check: bool = False,
                                convergence_range: int = 100) -> tuple[list[dict[int, float]], list[float]]:
    """
    Estimate state values by averaging returns from sampled episodes under the given policy.

    For each episode:
    - Generate an episode using the policy from a random starting state
    - For each state in the episode, calculate the return (discounted cumulative reward)
    - Track the return for each state-visitation
    - After all episodes, estimate each state's value as the average of all returns for that state

    Args:
        policy (dict[int, int]): Mapping from state index to selected action.
        num_of_episodes (int): Number of episodes to generate for estimation.
        gamma (float): Discount factor for future returns.
        convergence_check (bool, optional): Whether to check for convergence. Defaults to False.
        convergence_range (int, optional): Minimum number of iterations to converge. Defaults to 100.

    Returns:
        dict[int, float]: Mapping from state index to estimated value (average return from policy).
        if track_reward:
        list[float]: Cumulative reward for each episode.
    """
    # Initialize a dictionary to store all returns (G) for each state across episodes
    returns: dict[int, list[float]] = {
        state: [] for state in range(MouseEnv.num_of_states)
    }
    values: dict[int, float] = {}
    values_list: list[dict[int, float]] = []
    # Add list to store cumulative reward in
    list_rewards: list[float] = []
    # Generate and process multiple episodes
    if convergence_check:
        old_policy: dict[int, int] = {}
        convergence_counter: int = 0
    for _ in range(num_of_episodes):
        # Generate a single episode following the current policy
        episode = generate_episode(policy)
        # Initialize the return accumulator
        g = 0
        # Add variable to track total reward
        cumulative_reward = 0
        # Track visited states for first-visit Monte Carlo
        visited_states = [state[0] for state in episode]

        # Work backward through the episode to compute returns (backward iteration for discount efficiency)
        while episode:
            # Pop the last (state, action, reward) tuple from the episode
            state, action, reward = episode.pop()
            # Update the total reward
            cumulative_reward += reward
            # Update return using discounted reward: G = R + γ*G
            g = gamma * g + reward
            # For first-visit only: store the return if this is the first visit to the state in this episode
            if state not in visited_states[: len(episode)]:
                returns[state].append(g)
        # Add reward to rewards list
        list_rewards.append(cumulative_reward)

        # Compute state values as the average of collected returns
        for state in range(MouseEnv.num_of_states):
            if returns[state]:
                # Average the returns collected for this state
                values[state] = sum(returns[state]) / len(returns[state])
            else:
                # Unvisited states have zero value estimate
                values[state] = 0.0
        values_list.append(values.copy())

        # Extra convergence check for plotting
        if convergence_check:
            # Policy Improvement: Select greedy actions with respect to the estimated values
            new_policy = control(values_list[-1], gamma)
            # Convergence check: stop if the policy is stable (unchanged)
            if policy == old_policy:
                convergence_counter += 1
            else:
                convergence_counter = 0
            if convergence_counter >= convergence_range:
                break
            # Update to the improved policy and continue
            old_policy = new_policy.copy()

    return values_list, list_rewards

def track_montecarlo(num_of_episodes: int = 1000,
                     discount: float = 0.5,
                     convergence_check: bool = False,
                     convergence_range: int = 100) -> tuple[list[dict[int, float]], list[list[float]]]:
    """
    Run Monte Carlo control algorithm to find an optimal policy and state-value function.

    Alternates between policy evaluation (via Monte Carlo prediction) and policy improvement
    until the policy converges (remains unchanged across an iteration).

    Args:
        num_of_episodes (int): Number of episodes to generate during each policy evaluation step.
        discount (float): Discount factor for future returns.

    Returns:
        tuple:
            - list[dict[int, float]]: All state-value function mappings from state index to estimated value.
            - list[list[float]]: Final cumulative reward for each episode.
    """
    # Initialize policy randomly for all states
    policy = {
        state: random.randint(0, MouseEnv.num_of_actions - 1)
        for state in range(MouseEnv.num_of_states)
    }

    cumulative_rewards: list[list[float]] = []
    values_list: list[dict[int, float]] = []
    # Alternate between policy evaluation and policy improvement until convergence
    while True:
        # Policy Evaluation: Estimate state values under the current policy using Monte Carlo
        values, rewards = track_montecarlo_prediction(policy, num_of_episodes, discount,
                                                      convergence_check=convergence_check,
                                                      convergence_range=convergence_range)
        # Store rewards, values from all episodes in the lists
        cumulative_rewards.append(rewards)
        values_list.extend(values)
        # Policy Improvement: Select greedy actions with respect to the estimated values
        new_policy = control(values[-1], discount)
        # Convergence check: stop if the policy is stable (unchanged)
        if new_policy == policy:
            break
        # Update to the improved policy and continue
        policy = new_policy.copy()
    return values_list, cumulative_rewards