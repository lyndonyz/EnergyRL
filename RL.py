import numpy as np
from RoverMDP import RoverMDP

class RL:
    def __init__(self,mdp,sampleReward):
        """
        Constructor for the RL class

        Parameters:
            mdp: Markov decision process (T, R, discount)
            sampleReward: Function to sample rewards (e.g., bernoulli, Gaussian).
        """
        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        """
        Procedure to sample a reward and the next state\n
        reward ~ Pr(r)\n
        nextState ~ Pr(s'|s,a)

        Parameters:
            state (int): Current state.
            action (int): Action to be executed.

        Returns: 
            list: [Sampled reward, Sampled next state]
        """
        
        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        """
        Implementation of a Q-Learning algorithm.

        Parameters:
            s0 (int): Initial state for every episode.
            initialQ (np.ndarray): Initial Q-table.
            nEpisodes (int): Number of episodes to run for.
            nSteps (int): Number of steps in each episode.
            epsilon (float): Exploration rate for epsilon-greedy action selection. 0 by default.
            temperature (float): Boltzmann exploration rate. 0 by default.
        
        Returns:
            list: [Fully trained Q-table, optimal action for each state]
        """

        Q = initialQ.copy()
        N = np.zeros_like(Q)
        self.episodeRewards = [] 

        # Main logic loop
        for episode in range(nEpisodes):
            state = s0
            cumulativeReward = 0

            # Single episode loop
            for step in range(nSteps):
                # Random
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.mdp.nActions)
                else:
                    # Boltzmann exploration
                    if temperature > 0:
                        q_vals = Q[:, state]
                        exp_vals = np.exp(q_vals / temperature)
                        probs = exp_vals / np.sum(exp_vals)
                        action = np.random.choice(self.mdp.nActions, p=probs)
                    # Greedy
                    else:
                        action = np.argmax(Q[:, state])
                
                # Get reward and next state
                reward, nextState = self.sampleRewardAndNextState(state, action)
                cumulativeReward += (self.mdp.discount ** step) * reward

                # Update learning rate
                N[action, state] += 1
                alpha = 1.0 / N[action, state]

                # Update Q-value
                Q[action, state] += alpha * (
                    reward + self.mdp.discount * np.max(Q[:, nextState]) - Q[action, state]
                )
                state = nextState

            self.episodeRewards.append(cumulativeReward)
        
        # Return final Q function and policy
        policy = np.argmax(Q, axis=0)
        return [Q, policy]