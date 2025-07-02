# RL_Ex3
README.txt

Reinforcement Learning – Exercise 3
Students:
  Daniel Soref - 204798342 - soref@mail.tau.ac.il
  Shay Saraf - 326540721 - shaysaraf@mail.tau.ac.il

------------------------------------------------------------
Contents of Submission
------------------------------------------------------------

This submission includes:
1. RL_Course_Ex3.pdf — Written report containing answers and results for all questions (theory + practical).
2. control.py — Solution for Practical Question 1 (Off-Policy Model-Based control for CartPole).
3. cart_pole.py — Cart-pole simulator provided.
4. tabular_Q.py — Solution for Practical Question 2.1 (Tabular Q-Learning for FrozenLake).
5. network_Q.py — Solution for Practical Question 2.2 (Q-Learning with a neural network for FrozenLake).
6. README.txt — This file.

------------------------------------------------------------
Execution Instructions
------------------------------------------------------------

Each file can be executed independently using Python 3.12+. No specific package installation is required beyond `gym`, `numpy`, `torch`, and `matplotlib`.

———— Question 1: Off-Policy Model-Based ————
• Files: control.py, cart_pole.py
• To run:
    python3 control.py
• Output:
  - The number of trials it took to converge (reported in terminal).
  - A plot of the number of time steps the pole was balanced per trial.

———— Question 2.1: Tabular Q-Learning ————
• File: tabular_Q.py
• To run:
    python3 tabular_Q.py
• Output:
  - Displays training progress and prints final success rate and Q-table.
  - Supports both `is_slippery=True` and `is_slippery=False` via hardcoded env flags.

———— Question 2.2: Neural Network Q-Learning ————
• File: network_Q.py
• To run:
    python3 network_Q.py
• Output:
  - Trains a one-layer Q-network using a one-hot input state vector.
  - Prints final success rate and optionally plots learning curve.
  - Also supports both `is_slippery=True` and `is_slippery=False` via hardcoded env flags.

------------------------------------------------------------
Summary of Results (as described in PDF)
------------------------------------------------------------

• **Question 1 (Cart-Pole)**:
    - The algorithm converged after 172 trials.
    - Learning curve plotted in the PDF.

• **Question 2.1 (Tabular Q-Learning)**:
    - Performance depends on `is_slippery`.
    - Performs reasonably well for `is_slippery=False`.
    - Degrades with randomness (`is_slippery=True`), as expected.

• **Question 2.2 (Neural Network Q-Learning)**:
    - Performs worse than tabular Q-learning in deterministic (`is_slippery=False`) mode.
    - Outperforms tabular method in the slippery setting (`is_slippery=True`) due to its generalization ability.

------------------------------------------------------------
Note
------------------------------------------------------------
• Learning curves and results are also documented and embedded in the final PDF file as per submission guidelines.
