import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from environment import MouseEnv
from dp import policy_iteration
from mc import track_montecarlo
from td import track_SARSA

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
    unique_trajectories = len(
        {tuple(np.round(history, 10)) for history in value_history.values()}
    )
    total_tracked = len(value_history)
    _style_axis(ax, title, "Iteration", "Value")
    ax.text(
        0.02,
        0.98,
        f"states: {total_tracked}, unique trajectories: {unique_trajectories}",
        transform=ax.transAxes,
        va="top",
    )


def _plot_value_hist(ax, values: dict, title: str):
    ax.hist(list(values.values()), bins=100)
    _style_axis(ax, title, "Reward", "Frequency")


# Algorithm detail plotting function


def plot_algorithm_details(
    delta_history: list[float],
    value_history: dict[int, list[float]],
    values: dict,
    algorithm_name: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    _plot_delta(axes[0], delta_history, rf"{algorithm_name}: $\Delta$ per iteration")
    _plot_value_history(axes[1], value_history, f"{algorithm_name}: value history")
    _plot_value_hist(
        axes[2], values, f"{algorithm_name}: distribution of state rewards"
    )
    plt.tight_layout()
    return fig, axes


# Policy plotting function


def plot_policy(
    policy: dict, values: dict | None = None, title: str = "Policy by cheese position"
):
    if isinstance(values, np.ndarray):
        normalized_values = {}
        for state, action in policy.items():
            normalized_values[state] = float(values[state, action])
        values = normalized_values

    "Plot policy as action arrows, with optional value-based square coloring"
    ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    rows, cols = MouseEnv.rows, MouseEnv.cols
    position_policy = policy_index_to_positions(policy, rows, cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10), constrained_layout=True)

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
            ax.text(
                c_col,
                c_row,
                "C",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
            )
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

def plot_epsilon_decay(epsilon: float, epsilon_decay:float, episodes: int):
    X = np.arange(0, episodes)
    y = epsilon * epsilon_decay**X
    plt.plot(X, y)
    plt.title(f"Epsilon over {episodes} episodes, with a decay rate of {epsilon_decay}")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid()

def plot_cumulative_reward(num_of_episodes = 1000, discount = 0.5, alpha = 0.5, epsilon = 0.5, epsilon_decay = 0.99):
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
    list_of_rewards: list[list[float]]
    _, list_of_rewards = track_montecarlo(num_of_episodes, discount)

    midpoint = len(list_of_rewards) // 2
    mc_rewards = [list_of_rewards[0],
               list_of_rewards[midpoint],
               list_of_rewards[-1]]

    list_of_rewards: list[float]
    _, list_of_rewards2 = track_SARSA(num_of_episodes, alpha, discount, epsilon, epsilon_decay)

    first_point = len(list_of_rewards2) // 3
    second_point = 2 * len(list_of_rewards2) // 3
    td_rewards = [list_of_rewards2[:first_point],
                list_of_rewards2[first_point: second_point],
                list_of_rewards2[second_point:]]

    fig, axes = plt.subplots(2, 3, figsize=(13, 5))
    colors = ["red", "orange", "green"]
    mc_labels = ["First iteration",
                 f"Iteration {midpoint} (midpoint)",
                 "Last iteration"]
    for ax, reward, color, label in zip(axes[0], mc_rewards, colors, mc_labels):
        ax.scatter(range(len(reward)), reward, color=color, alpha=0.2, label="MC " + label)
        ax.hlines(sum(reward) / len(reward), xmin=0, xmax=len(reward), color=color,
                       linestyles="dashed")
        ax.set_ylim(-200, 100)


    td_labels = [f"First {first_point} episodes",
                 f"{first_point}-{second_point} episodes",
                 f"{second_point}-{len(list_of_rewards2)} episodes"]
    for ax, reward, color, label in zip(axes[1], td_rewards, colors, td_labels):
        ax.scatter(range(len(reward)), reward, color=color, alpha=0.2, label="TD " + label)
        ax.hlines(sum(reward) / len(reward), xmin=0, xmax=len(reward),
                  color=color, linestyles="dashed")
        ax.set_ylim(-200, 100)

    fig.suptitle("Cumulative Reward for episodes from the MC and TD Algorithms")
    fig.supxlabel("Episodes")
    fig.supylabel("Cumulative Reward")
    fig.legend()
    fig.show()

def plot_root_mean_squared_errors():
    """
    Plots the root-mean-square error for Monte Carlo and Temporal Difference.
    Note: the monte carlo plot is spiky because every batch of 1000 episodes starts with a new value set.
    """
    theta = 1e-5
    discount = 0.5
    num_of_episodes = 1000
    learning_rate = 0.5
    epsilon = 0.5
    epsilon_decay = 0.99

    _, dp_values = policy_iteration(theta=theta, gamma=discount)
    dp_values = list(dp_values.values())

    mc_list_values, _ = track_montecarlo(num_of_episodes, discount)
    mc_list_values = [list(mc_value.values()) for mc_value in mc_list_values]

    SARSA_qtables, _ = track_SARSA(num_of_episodes, learning_rate, discount, epsilon, epsilon_decay)
    td_list_values = [list(np.max(q_table, axis=1)) for q_table in SARSA_qtables]

    mc_r = [float(np.mean(np.power(np.subtract(dp_values, mc_values), 2)))
            for mc_values in mc_list_values]
    td_r = [float(np.mean(np.power(np.subtract(dp_values, td_values), 2)))
            for td_values in td_list_values]

    shared_max: float = max(max(mc_r), max(td_r))
    shared_max *= 1.1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(range(len(mc_r)), mc_r, color="blue", label="Monte Carlo")
    axes[0].set_ylim(0, shared_max)
    axes[1].plot(range(len(td_r)), td_r, color="green", label="Temporal Difference")
    axes[1].set_ylim(0, shared_max)
    fig.suptitle("Mean squared error compared to policy iteration")
    fig.supxlabel("Episodes")
    fig.supylabel("Mean squared error")
    fig.legend()
    fig.show()

if __name__ == "__main__":
    plot_cumulative_reward()
    plot_root_mean_squared_errors()