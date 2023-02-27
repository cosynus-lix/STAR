# GARA
This repository contains the code for Goal Abstraction via Reachability Analysis (GARA).
The code can be easily tested in Docker with the provided image.

This version implements the basic version of GARA in Maze environments and allows the 
comparison with hDQN  (with Handcrafted Representation), Feudal HRL that either samples
goals directly from the state space and learns, or that can learn a latent representation
for states using a LSTM network.