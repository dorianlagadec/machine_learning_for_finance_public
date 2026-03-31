GAIN_MATRIX = [[(3,3),(0,4)], # Payoff matrix: GAIN_MATRIX[my_action][opp_action] = (my_gain, opp_gain)
               [(4,0), (1,1)]]

ACTIONS_INDEX = {"C" : 0,
                 "B" : 1} # Index for actions

NUM_PLAYERS = 2 # Number of players in the game

NUM_TURNS = 100000 # Number of turns per matchup

P_COOP = 0.7 # Probability of cooperation for ProbaCooperation strategy

JOSS_P = 0.1 # Probability of betrayal for Joss strategy

