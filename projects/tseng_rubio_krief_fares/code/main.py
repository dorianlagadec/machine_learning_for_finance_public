from classes.game import Game
import default_params
import argparse
from strategies import *
from utils import *

def main():

    # Easier to execute the code with custom params from the terminal
    parser = argparse.ArgumentParser(description="Simulation params")

    parser.add_argument(
        "--nb_players",
        type=int,
        default=None,
        help="Number of players (default: from default_params)"
    )
    parser.add_argument(
        "--nb_turns",
        type=int,
        default=None,
        help="Number of turns (default: from default_params)"
    )

    args = parser.parse_args()

    num_players = args.nb_players if args.nb_players is not None else default_params.NUM_PLAYERS
    num_turns   = args.nb_turns   if args.nb_turns   is not None else default_params.NUM_TURNS

    # Here insert the strategy mix as a proportion of the total number of players
    strategy_mix = {
    TitForTat:          0,
    TwoTitsForTat:      0,
    TitForTwoTats:      0,
    SuspiciousTitForTat: 0,
    AlwaysCooperate:    0,
    AlwaysBetray:       0,
    RandomAction:       0,
    Joss:               0,
    DiscriminatingAltruist: 0,
    Bully:              0,
    ProbaCooperation:   0,
    
    QLearningStrategy: 1
}
    # Q-Learning hyperparameters
    ql_params = {
        "alpha"         : 0.1,    # Learning rate
        "gamma"         : 0.9,    # Discount factor — high weight on future rewards
        "epsilon"       : 1.0,     # Initial exploration rate (random at first)
        "epsilon_min"   : 0.01,    # Minimum exploration rate
        "epsilon_decay" : 0.9999, # ER decay 
    }

    # Play the game
    print(f"\nSimulation with {num_players} players who will play against each other {'{:.0f}'.format(num_turns)} times !\n")
    game = Game(num_players=num_players, num_turns=num_turns, strategy_mix=strategy_mix, ql_params=ql_params)
    game.play()
    plot_convergence(game, num_turns)



if __name__ == "__main__":
    main()
    #plot_epsilon_decay()