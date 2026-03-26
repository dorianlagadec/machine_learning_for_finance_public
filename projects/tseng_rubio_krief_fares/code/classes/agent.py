"""

The agent class will represent a person/player in the game that will interact with others.
Therefore, the class will instanciate this person and its actions during the game.
Each player has a certain strategy wether it be a predefined one, or a Q-learning one.
At each step of the game, in order to determine the output action, each player can access its 
previous relations with the other player. For each action, a decision will be sent to the game class
that will determine the score resulting of the interaction and record it in its history.

"""

from strategies import Strategy

class Agent:
    def __init__(self, id : str, strategy : Strategy):
        
        self.id = id # Unique identification for each player
        self.strategy = strategy  # Already-instantiated strategy object

        # At first, the agent met no one. Then store each result of its interactions with other agent. (Only access to its own results)
        self.interactions = {} 
        self.score = 0 # Null score at the beginning (consider starting with non null)
    
    def update_score(self, change : int) -> None :
        """ After an interaction with another agent, update its score 
        """
        self.score += change
    
    def update_interactions(self, other_agent_id : str, actions : tuple) -> None :
        """ After facing another player in the game, update each player interaction history. Acts like memory.

        Args:
            other_agent_id (str): id of the facing player
            actions (tuple): result action of the game turn. (Your choice, Opponent choice)
        """
        # Create history if player not met before
        if not other_agent_id in self.interactions :
            self.interactions[other_agent_id] = []
        
        self.interactions[other_agent_id].append(actions) # Update history

    def choose_action(self, other_agent_id) -> str :
        """ Core function of the class, given the opponent, its possible previous interactions and the player strategy,
        outputs the action of the player.

        Args:
            other_agent_id (_type_): id of the facing player

        Returns:
            str: "C" for cooperation or "B" for betrayal
        """
        return self.strategy.choose_action(self.id, other_agent_id, self.interactions)
    
    def get_score(self) -> int:
        """Return the agent's cumulative score."""
        return self.score
    
    def get_id(self) -> str :
        """Return the agent's id."""
        return self.id
    
    def get_strategy(self) :
        """Return the agent's strategy (as an object)."""
        return self.strategy
    
    def get_interactions(self) -> dict :
        """Return the agent's interactions (all)."""
        return self.interactions