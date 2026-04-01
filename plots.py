import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from environment import MouseEnv

# Plotting helper functions


def policy_index_to_positions(policy: dict, rows: int, cols: int):
    "Change policy dict with state indices as keys to a 4D array indexed by cheese and mouse positions"
    position_policy = np.full((rows, cols, rows, cols), np.nan)
    for state, action in policy.items():
        if MouseEnv.is_terminal_obs(state):
            continue
        mouse_pos, cheese_pos, won = MouseEnv.get_state_from_obs(state)
        m_row, m_col = mouse_pos  # type: ignore
        c_row, c_col = cheese_pos  # type: ignore
        position_policy[c_row, c_col, m_row, m_col] = action
    return position_policy


def _style_axis(ax, title: str, xlabel: str, ylabel: str, grid_axis: str = "both"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid_axis == "both":
        ax.grid(True)
    else:
        ax.grid(axis=grid_axis, alpha=0.3)


def _get_ax(ax, figsize=(6, 4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False


# Algorithm detail plotting helper functions


def _plot_delta(ax, history: list[float], title: str):
    ax.plot(range(len(history)), history)
    _style_axis(ax, title, "Iteration", r"$\Delta$")


def _plot_value_history(ax, value_history: dict[int, list[float]], title: str):
    for history in value_history.values():
        ax.plot(range(len(history)), history, alpha=0.75, linewidth=1)
    unique_trajectories = len({tuple(np.round(history, 10)) for history in value_history.values()})
    total_tracked = len(value_history)
    _style_axis(ax, title, "Iteration", "Value")
    ax.text(0.02, 0.98, f"states: {total_tracked}, unique trajectories: {unique_trajectories}", transform=ax.transAxes, va="top",)


def _plot_value_hist(ax, values: dict, title: str):
    ax.hist(list(values.values()), bins=100)
    _style_axis(ax, title, "Reward", "Frequency")


def _build_value_history(value_snapshots: list[dict[int, float]]) -> dict[int, list[float]]:
    if not value_snapshots:
        return {}
    states = sorted({state for snapshot in value_snapshots for state in snapshot.keys()})
    return {
        state: [snapshot.get(state, 0.0) for snapshot in value_snapshots]
        for state in states
    }


def _compute_delta_history(value_snapshots: list[dict[int, float]]) -> list[float]:
    if len(value_snapshots) < 2:
        return []

    delta_history: list[float] = []
    previous = value_snapshots[0]
    for current in value_snapshots[1:]:
        states = set(previous.keys()) | set(current.keys())
        delta = max(abs(current.get(state, 0.0) - previous.get(state, 0.0)) for state in states)
        delta_history.append(float(delta))
        previous = current
    return delta_history


# Algorithm detail plotting function


def plot_algorithm_details(
    delta_history: list[float],
    value_history: dict[int, list[float]],
    values: dict,
    algorithm_name: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    _plot_delta(axes[0], delta_history,
                rf"{algorithm_name}: $\Delta$ per iteration")
    _plot_value_history(axes[1], value_history,
                        f"{algorithm_name}: value history")
    _plot_value_hist(
        axes[2], values, f"{algorithm_name}: distribution of state rewards"
    )
    plt.tight_layout()
    return fig, axes


def plot_montecarlo_details(value_snapshots: list[dict[int, float]]):
    """Create DP-style detail plots for Monte Carlo value learning."""
    if not value_snapshots:
        raise ValueError("Monte Carlo produced no value snapshots")

    delta_history = _compute_delta_history(value_snapshots)
    value_history = _build_value_history(value_snapshots)
    final_values = value_snapshots[-1]
    return plot_algorithm_details(delta_history, value_history, final_values, "Monte Carlo")


def plot_td_details(q_tables: list[np.ndarray], algorithm_name: str = "Temporal Difference (SARSA)"):
    """Create DP-style detail plots for Temporal Difference learning from tracked Q snapshots."""
    if not q_tables:
        raise ValueError(f"{algorithm_name} produced no Q-table snapshots")

    value_snapshots = [
        {state: float(np.max(q_table[state])) for state in range(q_table.shape[0])}
        for q_table in q_tables
    ]
    delta_history = _compute_delta_history(value_snapshots)
    value_history = _build_value_history(value_snapshots)
    final_values = value_snapshots[-1]
    return plot_algorithm_details(delta_history, value_history, final_values, algorithm_name)


def plot_qlearning_details(q_tables: list[np.ndarray]):
    """Create DP-style detail plots for Temporal Difference (Q-learning) learning."""
    return plot_td_details(q_tables, algorithm_name="Temporal Difference (Q-learning)")


def plot_deep_qlearning_details(q_tables: list[np.ndarray]):
    """Create DP-style detail plots for Deep Q-learning value learning."""
    return plot_td_details(q_tables, algorithm_name="Deep Q-learning")


# Policy plotting function


def plot_policy(policy: dict, values: dict | np.ndarray | None = None, title: str = "Policy by cheese position"):
    if isinstance(values, np.ndarray):
        normalized_values = {}
        for state, action in policy.items():
            normalized_values[state] = float(values[state, action])
        values = normalized_values

    "Plot policy as action arrows, with optional value-based square coloring"
    ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    rows, cols = MouseEnv.rows, MouseEnv.cols
    position_policy = policy_index_to_positions(policy, rows, cols)
    fig, axes = plt.subplots(rows, cols, figsize=(
        16, 10), constrained_layout=True)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "value_map", ["red", "yellow", "green"]
    )
    norm = mcolors.TwoSlopeNorm(
        vmin=MouseEnv.lose_punishment, vcenter=0, vmax=MouseEnv.win_reward
    )

    for c_row, c_col in np.ndindex(rows, cols):
        ax = axes[c_row, c_col]
        ax.set(xlim=(-0.5, cols - 0.5), ylim=(rows - 0.5, -0.5))
        ax.set_xticks([])
        ax.set_yticks([])

        # Terminal cheese locations are end states, so no action arrows are shown
        if c_col == 0 or c_row == rows - 1:
            ax.add_patch(plt.Rectangle((c_col - 0.5, c_row - 0.5), 1, 1, facecolor="lightpink"))  # type: ignore
            ax.text(c_col, c_row, "C", ha="center", va="center", fontsize=20, fontweight="bold")
            continue

        # Draw colored backgrounds for each mouse position
        for m_row, m_col in np.ndindex(rows, cols):
            if (m_row, m_col) == (c_row, c_col):
                cell_color = "lightgray"
            elif values is None:
                cell_color = "white"
            else:
                state = MouseEnv.get_obs([m_row, m_col], [c_row, c_col], False)
                cell_color = cmap(norm(values[state]))
            ax.add_patch(plt.Rectangle((m_col - 0.5, m_row - 0.5), 1, 1, facecolor=cell_color))  # type: ignore

        # Draw cheese position marker
        ax.text(
            c_col, c_row, "C", ha="center", va="center", fontsize=20, fontweight="bold"
        )

        # Draw action arrows for each mouse position
        for m_row, m_col in np.ndindex(rows, cols):
            if (m_row, m_col) == (c_row, c_col):
                continue
            action = position_policy[c_row, c_col, m_row, m_col]
            if not np.isnan(action):
                ax.text(
                    m_col,
                    m_row,
                    ARROWS[int(action)],
                    ha="center",
                    va="center",
                    fontsize=25,
                )

    if values is not None:
        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        fig.colorbar(
            scalar_mappable,
            ax=axes.ravel().tolist(),
            shrink=0.7,
            pad=0.02,
            label="State value",
        )

    fig.suptitle(title)
    return fig, axes


def plot_discount_rate_policy_row(
    discount_rates: list[float],
    policies: list[dict],
    values_list: list[dict | np.ndarray],
    iterations: list[int] | None = None,
):
    """Plot policies for multiple discount rates side by side.

    Renders each policy figure to an image and arranges them in one horizontal row.
    """
    if len(discount_rates) == 0:
        raise ValueError("discount_rates must contain at least one value")
    if len(policies) != len(discount_rates) or len(values_list) != len(discount_rates):
        raise ValueError("discount_rates, policies, and values_list must have equal length")

    if iterations is None:
        iterations = [0] * len(discount_rates)
    elif len(iterations) != len(discount_rates):
        raise ValueError("iterations must have the same length as discount_rates")

    images: list[np.ndarray] = []

    for policy, values in zip(policies, values_list):
        policy_fig, _ = plot_policy(
            policy, values=values, title="")

        policy_fig.canvas.draw()
        width, height = policy_fig.canvas.get_width_height()
        image = np.frombuffer(policy_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape( # type: ignore
            height, width, 4)  # type: ignore
        plt.close(policy_fig)

        images.append(image)

    fig, axes = plt.subplots(1, len(discount_rates),
                             figsize=(8 * len(discount_rates), 7))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, discount, image, n_iter in zip(axes_array, discount_rates, images, iterations):
        ax.imshow(image)
        ax.set_title(f"gamma = {discount} | iters = {n_iter}")
        ax.axis("off")

    plt.tight_layout()
    return fig, axes


# Comparison plot functions


def plot_policy_agreement(
    policy_a: dict,
    policy_b: dict,
    label_a: str = "Policy A",
    label_b: str = "Policy B",
    ax=None,
):
    non_terminal_states = [
        s for s in range(MouseEnv.num_of_states) if not MouseEnv.is_terminal_obs(s)
    ]
    agreement = sum(policy_a[s] == policy_b[s] for s in non_terminal_states)
    disagreement = len(non_terminal_states) - agreement

    fig, ax, created_fig = _get_ax(ax, figsize=(6, 4))
    ax.bar(["Agreement", "Disagreement"], [agreement, disagreement])
    _style_axis(
        ax,
        f"Policy similarity: {label_a} vs {label_b}",
        "",
        "Number of non-terminal states",
        grid_axis="y",
    )

    if created_fig:
        return fig, ax
    return ax.figure, ax


def plot_value_function_comparison(
    values_a: dict,
    values_b: dict,
    label_a: str = "Values A",
    label_b: str = "Values B",
    ax=None,
):
    x = np.array([values_a[s] for s in sorted(values_a.keys())])
    y = np.array([values_b[s] for s in sorted(values_b.keys())])
    min_v = min(x.min(), y.min())
    max_v = max(x.max(), y.max())

    fig, ax, created_fig = _get_ax(ax, figsize=(6, 6))
    ax.scatter(x, y, alpha=0.6, s=16)
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    _style_axis(ax, "State-value comparison", label_a, label_b)

    if created_fig:
        return fig, ax
    return ax.figure, ax


def plot_dp_comparisons(
    final_policy: dict,
    value_policy: dict,
    policy_iteration_values: dict,
    value_iteration_values: dict,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_policy_agreement(
        final_policy,
        value_policy,
        label_a="Policy Iteration",
        label_b="Value Iteration",
        ax=axes[0],
    )
    plot_value_function_comparison(
        policy_iteration_values,
        value_iteration_values,
        label_a="Policy Iteration state values",
        label_b="Value Iteration state values",
        ax=axes[1],
    )
    plt.tight_layout()
    return fig, axes


def plot_epsilon_decay(epsilon: float, epsilon_decay: float, episodes: int):
    X = np.arange(0, episodes)
    y = epsilon * epsilon_decay**X
    plt.plot(X, y)
    plt.title(
        f"Epsilon over {episodes} episodes, with a decay rate of {epsilon_decay}")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid()


def plot_cumulative_reward(num_of_episodes=1000,
                           mc_reward_batches: list[list[float]] | None = None,
                           td_reward_history: list[float] | None = None):
    """
    Plots the cumulative reward for Monte Carlo and Temporal Difference at three different stages:
    - At the start, where no learning has happened.
    - Midway the learning process.
    - At the end, where the full learning has happened.

    Note: The TD algorithm greatly outperforms the amount of episodes computed, so less would be necessary.

    Args:
        num_of_episodes (int): the amount of episodes for MC and TD.
        discount (float) the discount factor, devaluing states further in the future [0, 1] -> the higher
        alpha (float): the learning rate.
        epsilon (float): the degree of exploration [0, 1] -> the higher, the more random.
        epsilon_decay (float): the base of decay of exploration [0, 1] -> the higher, the slower the decay.
    """
    if mc_reward_batches is None or td_reward_history is None:
        raise ValueError("Pass mc_reward_batches and td_reward_history computed in the notebook")
    list_of_rewards = mc_reward_batches

    midpoint = len(list_of_rewards) // 2
    mc_rewards = [list_of_rewards[0],
                  list_of_rewards[midpoint],
                  list_of_rewards[-1]]

    list_of_rewards2 = td_reward_history

    first_point = len(list_of_rewards2) // 3
    second_point = 2 * len(list_of_rewards2) // 3
    td_rewards = [list_of_rewards2[:first_point],
                  list_of_rewards2[first_point: second_point],
                  list_of_rewards2[second_point:]]

    fig, axes = plt.subplots(2, 3, figsize=(13, 5))
    colors = ["red", "orange", "green"]
    mc_labels = ["First 1000 episodes",
                 f"#{midpoint} 1000 episodes (midpoint)",
                 "Last 1000 episodes"]
    for ax, reward, color, label in zip(axes[0], mc_rewards, colors, mc_labels):
        ax.scatter(range(len(reward)), reward, color=color, # type: ignore
                   alpha=0.2, label="MC " + label)
        ax.hlines(sum(reward) / len(reward), xmin=0, xmax=len(reward), color=color, # type: ignore
                  linestyles="dashed")
        ax.set_ylim(-200, 100)

    td_labels = [f"First {first_point} episodes",
                 f"{first_point}-{second_point} episodes",
                 f"{second_point}-{len(list_of_rewards2)} episodes"]
    for ax, reward, color, label in zip(axes[1], td_rewards, colors, td_labels):
        ax.scatter(range(len(reward)), reward, color=color,
                   alpha=0.2, label="TD " + label)
        ax.hlines(sum(reward) / len(reward), xmin=0, xmax=len(reward),
                  color=color, linestyles="dashed")
        ax.set_ylim(-200, 100)

    fig.suptitle(
        "Cumulative Reward for episodes from the MC and TD Algorithms")
    fig.supxlabel("Episodes")
    fig.supylabel("Cumulative Reward")
    fig.legend()
    plt.show()


def plot_root_mean_squared_errors(theta=1e-5,
                                  dp_values: dict[int, float] | None = None,
                                  mc_value_snapshots: list[dict[int, float]] | None = None,
                                  td_qtables: list[np.ndarray] | None = None):
    """
    Plots the root-mean-square error for Monte Carlo and Temporal Difference.

    Note: the monte carlo plot is spiky because every batch of 1000 episodes starts with a new value set.
    """

    if dp_values is None or mc_value_snapshots is None or td_qtables is None:
        raise ValueError("Pass dp_values, mc_value_snapshots, and td_qtables computed in the notebook")
    dp_values_list = list(dp_values.values())

    mc_list_values = mc_value_snapshots
    mc_list_values = [list(mc_value.values()) for mc_value in mc_list_values]

    td_list_values = [list(np.max(q_table, axis=1))
                      for q_table in td_qtables]

    mc_r = [float(np.mean(np.power(np.subtract(dp_values_list, mc_values), 2)))
            for mc_values in mc_list_values]
    td_r = [float(np.mean(np.power(np.subtract(dp_values_list, td_values), 2)))
            for td_values in td_list_values]

    shared_max: float = max(max(mc_r), max(td_r))
    shared_max *= 1.1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(range(len(mc_r)), mc_r, color="blue", label="Monte Carlo")
    axes[0].set_ylim(0, shared_max)
    axes[1].plot(range(len(td_r)), td_r, color="green",
                 label="Temporal Difference")
    axes[1].set_ylim(0, shared_max)
    fig.suptitle("Mean squared error compared to policy iteration")
    fig.supxlabel("Episodes")
    fig.supylabel("Mean squared error")
    fig.legend()
    plt.show()


def plot_sample_efficiency(iterations: int = 5,
                           mc_episode_numbers: list[int] | None = None,
                           td_episode_numbers: list[int] | None = None,
                           convergence_range: int = 100):
    """
    Plots the sample efficiency for Monte Carlo and Temporal Difference.

    Note: even just 10 iterations might already take a minute, don't set it too high.
    """

    if mc_episode_numbers is None or td_episode_numbers is None:
        raise ValueError("Pass mc_episode_numbers and td_episode_numbers computed in the notebook")

    iterations = min(len(mc_episode_numbers), len(td_episode_numbers))
    if iterations == 0:
        raise ValueError("mc_episode_numbers and td_episode_numbers must contain at least one value")

    mc_episode_numbers = mc_episode_numbers[:iterations]
    td_episode_numbers = td_episode_numbers[:iterations]

    mc_episode_number: int = int(
        sum(mc_episode_numbers) / len(mc_episode_numbers))
    td_episode_number: int = int(
        sum(td_episode_numbers) / len(td_episode_numbers))

    plt.bar(["Monte Carlo", "Temporal Difference"], [mc_episode_number, td_episode_number],
            color=["blue", "green"],
            label=[f"MC (mean={mc_episode_number} episodes)", f"TD (mean={td_episode_number} episodes)"])

    plt.ylabel("Episodes")
    plt.xlabel("Algorithm")
    plt.title(f"Plot of Sample Efficiency (mean of {iterations} iterations)\n"
              f"Terminates when policy remains the same in {convergence_range} episodes.")
    plt.hlines([mc_episode_number, td_episode_number], -1, 2,
               linestyles="dotted", color=["blue", "green"])
    plt.legend()
    plt.show()