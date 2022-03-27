# DDQL-Pacb
Double Deep Q-Learning Implementation for access control through Pacb adaptation in cellular networks

The MATLAB file entrenaDDQL1diaexp1.zip contains the first 200 results from the DDQL solution trained with 1 day of data and using a neural network with 10 hidden layers. In this experiment there are 30000 M2M users. These results were part of the journal paper "Deep reinforcement learning mechanism for dynamic access control in wireless networks handling mMTC" D. Pacheco-Paramo, L. Tello-Oquendo, V. Pla, J. Martinez Bauset. Ad Hoc Networks, November 2019.

The trained neural network Qnet in the file "entrena1diaDDQLexp1.mat" has 4 inputs (one for each state variable) and 16 outputs (one for each action). 


The original LTE simulator with access control was implemented by Luis Tello https://github.com/lptelloq/LTE-A_RACHprocedure

My contribution is an implementation of Double Deep Q-Learning for access control optimization in a system that can adapt Pacb using the LTE-A simulator
