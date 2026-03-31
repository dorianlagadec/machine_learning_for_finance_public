from abc import ABC, abstractmethod
import random
import default_params
from collections import defaultdict

class Strategy(ABC):
    """ Common interface for predefined strategies. (Like interface in Java)
    """

    @abstractmethod
    def choose_action(self, my_id: str, other_player_id: str, interactions: dict) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def update_Q(self, other_player_id: str, my_action: str, opp_action: str, reward: int) -> None:
        """Update Q-table for learning strategies. Default does nothing."""
        pass


class AlwaysCooperate(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """No matter what, cooperate.
        """
        return "C"

    def __str__(self) -> str:
        return "AlwaysCooperate"


class AlwaysBetray(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """No matter what, betrays.
        """
        return "B"

    def __str__(self) -> str:
        return "AlwaysBetray"


class RandomAction(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Takes a random choice at each interaction.
        """
        return random.choice(["C", "B"])

    def __str__(self) -> str:
        return "RandomAction"


class ProbaCooperation(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Cooperate with a fixed probability given in the default params.
        """
        return random.choices(["C", "B"], weights=[default_params.P_COOP, 1 - default_params.P_COOP])[0]

    def __str__(self) -> str:
        return f"ProbaCooperation(p={default_params.P_COOP})"

class TitForTat(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Cooperates on the first round, then imitates opponent's previous move.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) == 0:
            return "C"
        return interactions[other_player_id][-1]["opponent_action"]

    def __str__(self) -> str:
        return "TitForTat"

class SuspiciousTitForTat(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Defects on the first round, then imitates opponent's previous move.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) == 0:
            return "B"
        return interactions[other_player_id][-1]["opponent_action"]

    def __str__(self) -> str:
        return "SuspiciousTitForTat"

class TitForTwoTats(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Defects only if opponent defected twice in a row.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) < 2:
            return "C"
        last_two = [r["opponent_action"] for r in interactions[other_player_id][-2:]]
        return "B" if last_two == ["B", "B"] else "C"

    def __str__(self) -> str:
        return "TitForTwoTats"

class TwoTitsForTat(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Defects twice after being defected against, otherwise cooperates.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) == 0:
            return "C"
        last_two = [r["opponent_action"] for r in interactions[other_player_id][-2:]]
        if "B" in last_two:
            return "B"
        return "C"

    def __str__(self) -> str:
        return "TwoTitsForTat"

class DiscriminatingAltruist(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """In the Optional IPD, cooperates with any player that has never defected against it,
        otherwise refuses to engage.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) == 0:
            return "C"
        ever_defected = any(r["opponent_action"] == "B" for r in interactions[other_player_id])
        return "B" if ever_defected else "C"

    def __str__(self) -> str:
        return "DiscriminatingAltruist"
    
class Bully(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """Defects until opponent defects, then cooperates. Exploits overly accommodating strategies.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) == 0:
            return "B"
        last_opponent_action = interactions[other_player_id][-1]["opponent_action"]
        return "C" if last_opponent_action == "B" else "B"

    def __str__(self) -> str:
        return "Bully"

class Joss(Strategy):
    def choose_action(self, my_id, other_player_id, interactions) -> str:
        """TFT with occasional random defections to exploit cooperative opponents.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) == 0:
            return "C"
        last_opponent_action = interactions[other_player_id][-1]["opponent_action"]
        if last_opponent_action == "B":
            return "B"
        return random.choices(["C", "B"], weights=[1 - default_params.JOSS_P, default_params.JOSS_P])[0]

    def __str__(self) -> str:
        return f"Joss(p_betray={default_params.JOSS_P})"


class QLearningStrategy(Strategy):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9999):
        self.alpha = alpha              # Learning rate
        self.gamma = gamma              # Discount factor
        self.epsilon = epsilon          # Current exploration rate (starts high)
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Multiplicative decay factor per step
        self.q_table: dict = defaultdict(float)   # Q-table: (state, action) -> value
        self.last_state_action = {}         # Store last (state, action) per opponent

        # History for convergence plotting
        self.history = {"q_betray": [], "q_coop": [], "epsilon_hist": [], "action": []}

    def _get_state(self, interactions, other_player_id):
        """Get current state as (my_last_action, opp_last_action).
        Including the agent's own last action lets it learn that
        C->C leads to mutual cooperation, while B->B leads to punishment.
        """
        if other_player_id not in interactions or len(interactions[other_player_id]) == 0:
            return ()  # No history yet
        last = interactions[other_player_id][-1]
        return (last["player_action"], last["opponent_action"])  # (my last action, opp last action)

    def choose_action(self, my_id, other_player_id, interactions) -> str:
        state = self._get_state(interactions, other_player_id)
        actions = ["C", "B"]
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice(actions)
        else:
            q_values = [self.q_table[(state, a)] for a in actions]
            max_q = max(q_values)
            # If tie, choose randomly among max
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
            action = random.choice(best_actions)
        
        # Store for update
        self.last_state_action[other_player_id] = (state, action)

        # Record history for convergence plot
        ref = ("C", "C")
        self.history["q_betray"].append(self.q_table[(ref, "B")])
        self.history["q_coop"].append(self.q_table[(ref, "C")])
        self.history["epsilon_hist"].append(self.epsilon)
        self.history["action"].append(action)

        # Decay epsilon after each step
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action

    def update_Q(self, other_player_id: str, my_action: str, opp_action: str, reward: int) -> None:
        if other_player_id not in self.last_state_action:
            return
        old_state, action_taken = self.last_state_action[other_player_id]
        # new_state must match the format returned by _get_state:
        # (my_last_action, opp_last_action) — i.e. what just happened this turn.
        new_state = (my_action, opp_action) if old_state else ()
        
        # Q-learning update (Bellman equation)
        old_q = self.q_table[(old_state, action_taken)]
        max_next_q = max(self.q_table[(new_state, a)] for a in ["C", "B"]) if new_state else 0
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(old_state, action_taken)] = new_q

    def __str__(self) -> str:
        return f"QLearningStrategy(alpha={self.alpha}, gamma={self.gamma}, epsilon_start=1.0, epsilon_min={self.epsilon_min}, decay={self.epsilon_decay})"