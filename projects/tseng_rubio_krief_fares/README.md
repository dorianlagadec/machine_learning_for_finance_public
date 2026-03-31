# Reinforcement Learning in the Iterated Prisoner's Dilemma

## Authors

- Gabriel Tseng
- Axel Rubio
- Benjamin Krief
- Krea Fares

## Focus

The project explores the application of Q-learning to the Iterated Prisoner's Dilemma (IPD). The main objective is to understand how an adaptive agent learns to navigate strategic interactions without prior knowledge of the game structure.

Key areas of study include:
- The behavior of a rational learning agent against classical deterministic strategies (both naive and reactive).
- The analysis of conditions under which cooperation or defection emerges.
- The relative importance of Q-learning hyperparameters: learning rate, discount factor, and the exploration policy.

## Project Structure

- `main.py`: Main execution script to run the simulations.
- `strategies.py`: Implementation of the various baseline and reactive strategies (e.g., Tit For Tat, Always Defect, Random Action).
- `utils.py` / `plot_*.py`: Utility functions and plotting scripts for analyzing the simulation results and the impact of hyperparameters.
- `classes/`: Core object-oriented components defining the game environment and agents.
- `rapport.tex`: LaTeX source code for the comprehensive final report detailing the methodology and experimental results.


