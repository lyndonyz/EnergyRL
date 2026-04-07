import numpy as np
from RoverMDP import RoverMDP

class RL:
    def __init__(self,mdp,sampleReward):
        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        Q = initialQ.copy()
        N = np.zeros_like(Q)
        self.episodeRewards = [] 

        for episode in range(nEpisodes):
            state = s0
            cumulativeReward = 0
            for step in range(nSteps):
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.mdp.nActions)
                else:
                    if temperature > 0:
                        q_vals = Q[:, state]
                        exp_vals = np.exp(q_vals / temperature)
                        probs = exp_vals / np.sum(exp_vals)
                        action = np.random.choice(self.mdp.nActions, p=probs)
                    else:
                        action = np.argmax(Q[:, state])
                reward, nextState = self.sampleRewardAndNextState(state, action)
                cumulativeReward += (self.mdp.discount ** step) * reward
                N[action, state] += 1
                alpha = 1.0 / N[action, state]
                Q[action, state] += alpha * (
                    reward + self.mdp.discount * np.max(Q[:, nextState]) - Q[action, state]
                )
                state = nextState
            self.episodeRewards.append(cumulativeReward)
        policy = np.argmax(Q, axis=0)
        return [Q, policy]