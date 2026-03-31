import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import numpy as np
import os
import default_params

from classes.game import Game
from strategies import *


# ─────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────

def _run_ql_game(strategy_mix, ql_params, num_turns):
    """Run a 2-player game and return the QL agent's action history."""
    game = Game(num_players=2, num_turns=num_turns,
                strategy_mix=strategy_mix, ql_params=ql_params)
    game.play()
    ql_agent = next(
        a for a in game.players.values()
        if isinstance(a.get_strategy(), QLearningStrategy)
    )
    return ql_agent.get_strategy().history


def _rolling_coop(hist, window=None):
    """Return (turns_roll, coop_rate) arrays from an action history dict."""
    actions_bin = np.array([1 if a == "C" else 0 for a in hist["action"]], dtype=float)
    T = len(actions_bin)
    w = window or max(200, T // 100)
    coop_rate = np.convolve(actions_bin, np.ones(w) / w, mode="valid")
    turns_roll = np.arange(w, T + 1)
    return turns_roll, coop_rate, w


# ─────────────────────────────────────────────
# 1. Alpha impact – TwoTitsForTat
# ─────────────────────────────────────────────

def plot_alpha_impact_ttft(num_turns=50_000, output_path="img/alpha_impact_ttft.png"):
    """
    Plot the impact of alpha on QL convergence against TwoTitsForTat.
    """
    alphas = [0.001, 0.01, 0.1, 0.5, 0.9, 1]
    colors = ['#E63946', '#F4A261', '#E9C46A', '#2A9D8F', '#264653', '#000000']

    fig, ax = plt.subplots(figsize=(10, 6))
    window = None

    for alpha, color in zip(alphas, colors):
        print(f"\nRunning alpha={alpha}")
        ql_params = {
            "alpha": alpha, "gamma": 0.9,
            "epsilon": 1.0, "epsilon_min": 0.01,
            "epsilon_decay": 0.9999,  # fast decay to isolate alpha's effect
        }
        hist = _run_ql_game({TwoTitsForTat: 0.5, QLearningStrategy: 0.5},
                            ql_params, num_turns)
        turns_roll, coop_rate, window = _rolling_coop(hist)
        ax.plot(turns_roll, coop_rate, label=f"α = {alpha}", color=color, lw=1.5)

    ax.set_title("Impact of Learning Rate (α) on convergence against TwoTitsForTat (50K turns)")
    ax.set_xlabel("Turn")
    ax.set_ylabel(f"Cooperation rate (window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {os.path.abspath(output_path)}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 2. Alpha impact – TitForTat
# ─────────────────────────────────────────────

def plot_alpha_impact_tft(num_turns=50_000, output_path="img/alpha_impact_tft.png"):
    """
    Plot the impact of alpha on QL convergence against TitForTat.
    """
    alphas = [0.001, 0.01, 0.1, 0.5, 0.9, 1]
    colors = ['#E63946', '#F4A261', '#E9C46A', '#2A9D8F', '#264653', '#000000']

    fig, ax = plt.subplots(figsize=(10, 6))
    window = None

    for alpha, color in zip(alphas, colors):
        print(f"\nRunning alpha={alpha}")
        ql_params = {
            "alpha": alpha, "gamma": 0.9,
            "epsilon": 1.0, "epsilon_min": 0.01,
            "epsilon_decay": 0.9999,
        }
        hist = _run_ql_game({TitForTat: 0.5, QLearningStrategy: 0.5},
                            ql_params, num_turns)
        turns_roll, coop_rate, window = _rolling_coop(hist)
        ax.plot(turns_roll, coop_rate, label=f"α = {alpha}", color=color, lw=1.5)

    ax.set_title("Impact of Learning Rate (α) on convergence against TitForTat (50K turns)")
    ax.set_xlabel("Turn")
    ax.set_ylabel(f"Cooperation rate (window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {os.path.abspath(output_path)}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 3. Naive strategies
# ─────────────────────────────────────────────

def plot_naive_coop(num_turns=100_000, output_path="img/naive_coop.png"):
    """
    Plot the impact of alpha on QL convergence against naive strategies.
    """
    strats = [
        (AlwaysCooperate, "AlwaysCooperate", '#2A9D8F'),
        (AlwaysBetray,    "AlwaysBetray",    '#E63946'),
        (RandomAction,    "RandomAction",    '#F4A261'),
        (ProbaCooperation,"ProbaCooperation",'#264653'),
    ]
    base_ql = {
        "alpha": 0.1, "gamma": 0.9,
        "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.9999,
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    window = None

    for strat_class, strat_name, color in strats:
        print(f"\nRunning QLearning vs {strat_name}")
        hist = _run_ql_game({strat_class: 0.5, QLearningStrategy: 0.5},
                            base_ql, num_turns)
        turns_roll, coop_rate, window = _rolling_coop(hist)
        ax.plot(turns_roll, coop_rate, label=strat_name, color=color, lw=1.5)

    ax.set_title("Q-Learning Cooperation Rate vs Naive Strategies")
    ax.set_xlabel("Turn")
    ax.set_ylabel(f"Cooperation rate (window={window})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {os.path.abspath(output_path)}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 4. Lenient TFT variants
# ─────────────────────────────────────────────

def plot_lenient_tft_coop(num_turns=100_000, output_path="img/lenient_tft_coop.png"):
    """
    Plot the impact of alpha on QL convergence against lenient TFT variants.
    """
    strats = [
        (TitForTat,     "TitForTat",     '#E63946'),
        (TitForTwoTats, "TitForTwoTats", '#F4A261'),
    ]
    base_ql = {
        "alpha": 0.1, "gamma": 0.9,
        "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.9999,
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    window = None

    for strat_class, strat_name, color in strats:
        print(f"\nRunning QLearning vs {strat_name}")
        hist = _run_ql_game({strat_class: 0.5, QLearningStrategy: 0.5},
                            base_ql, num_turns)
        turns_roll, coop_rate, window = _rolling_coop(hist)
        ax.plot(turns_roll, coop_rate, label=strat_name, color=color, lw=1.5)

    ax.set_title("Q-Learning Cooperation Rate vs Lenient Tit-for-Tat")
    ax.set_xlabel("Turn")
    ax.set_ylabel(f"Cooperation rate (window={window})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {os.path.abspath(output_path)}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 5. Gamma impact
# ─────────────────────────────────────────────

def plot_gamma_impact(num_turns=100_000, output_path="img/gamma_impact.png"):
    """
    Plot the impact of gamma on QL convergence against TitForTat.
    """
    gammas = [0.1, 0.2, 0.3, 1/3, 0.4, 0.6, 0.8, 0.99]
    colors = cm.coolwarm(np.linspace(1, 0, len(gammas)))

    fig, ax = plt.subplots(figsize=(10, 6))
    window = None

    for gamma, color in zip(gammas, colors):
        print(f"\nRunning gamma={gamma:.3f}")
        ql_params = {
            "alpha": 0.1, "gamma": gamma,
            "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.9999,
        }
        hist = _run_ql_game({TitForTat: 0.5, QLearningStrategy: 0.5},
                            ql_params, num_turns)
        turns_roll, coop_rate, window = _rolling_coop(hist)

        if abs(gamma - 1/3) < 1e-5:
            ax.plot(turns_roll, coop_rate, label="γ = 1/3 (Threshold)",
                    color='black', lw=2, linestyle='--')
        else:
            ax.plot(turns_roll, coop_rate, label=f"γ = {gamma:.2f}",
                    color=color, lw=1.5)

    ax.set_title("Impact of Discount Factor (γ) on convergence against TitForTat (100K turns)")
    ax.set_xlabel("Turn")
    ax.set_ylabel(f"Cooperation rate (window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {os.path.abspath(output_path)}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 6. Epsilon decay impact
# ─────────────────────────────────────────────

def plot_epsilon_decay(num_turns=100_000, output_path="img/epsilon_decay.png"):
    """
    Plot the impact of epsilon decay on QL convergence against TitForTat.
    """
    if num_turns is None:
        num_turns = default_params.NUM_TURNS

    decays = [0.999, 0.9999, 0.99999]
    colors = ['#E63946', '#2A9D8F', '#E9C46A']

    fig, ax = plt.subplots(figsize=(10, 6))
    window = None

    for decay, color in zip(decays, colors):
        print(f"\nRunning decay={decay}")
        ql_params = {
            "alpha": 0.1, "gamma": 0.9,
            "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": decay,
        }
        hist = _run_ql_game({TitForTat: 0.5, QLearningStrategy: 0.5},
                            ql_params, num_turns)
        turns_roll, coop_rate, window = _rolling_coop(hist)
        ax.plot(turns_roll, coop_rate, label=f"decay = {decay}", color=color, lw=1.5)

    ax.set_title("Impact of Exploration Decay Rate (ε) on Cooperation against TitForTat")
    ax.set_xlabel("Turn")
    ax.set_ylabel(f"Cooperation rate (window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {os.path.abspath(output_path)}")
    plt.close(fig)

def plot_convergence(game, num_turns: int, output_path: str = "img/convergence.png") -> None:
    """
    Generate and save a single-panel figure illustrating the convergence of the
    Q-learning agent, showing the rolling cooperation rate alongside the exploration rate.
    """
    ql_agents = [
        agent for agent in game.players.values()
        if isinstance(agent.get_strategy(), QLearningStrategy)
    ]
    if not ql_agents:
        print("No Q-learning agent found; figure not generated.")
        return

    strat = ql_agents[0].get_strategy()
    hist  = strat.history
    T     = len(hist["action"])
    turns = np.arange(1, T + 1)

    window = max(200, T // 100)
    coop_bin   = np.array([1 if a == "C" else 0 for a in hist["action"]], dtype=float)
    coop_rate  = np.convolve(coop_bin, np.ones(window) / window, mode="valid")
    turns_roll = turns[window - 1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        "Convergence of the Q-agent — cooperation rate\n"
        f"(α={strat.alpha}, γ={strat.gamma}, "
        f"ε_min={strat.epsilon_min}, decay={strat.epsilon_decay})",
        fontsize=13, fontweight="bold"
    )

    ax.plot(turns_roll, coop_rate, color="steelblue", lw=1.5,
            label=f"Cooperation rate (window={window})")
    ax.axhline(0.0, color="steelblue", lw=0.8, ls="--", alpha=0.5,
               label="Full defection baseline")
    ax.set_ylabel("Cooperation rate", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Turn")
    ax.set_title("Agent behaviour over time")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(turns, hist["epsilon_hist"], color="tomato", lw=1, alpha=0.7,
             label="ε (exploration rate)")
    ax2.set_ylabel("ε", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax2.set_ylim(-0.05, 1.15)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {os.path.abspath(output_path)}")
    plt.close()