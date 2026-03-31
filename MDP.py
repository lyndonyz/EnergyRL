import numpy as np

class MDP:
    def __init__(self, T, R, discount):
        assert T.ndim == 3
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions, self.nStates, self.nStates)
        assert (abs(T.sum(2) - 1) < 1e-5).all()
        self.T = T

        assert R.ndim == 2
        assert R.shape == (self.nActions, self.nStates)
        self.R = R

        assert 0 <= discount < 1
        self.discount = discount

    def valueIteration(self, initialV, nIterations=np.inf, tolerance=0.01):
        V = initialV
        iterId = 0
        epsilon = np.inf

        while iterId < nIterations and epsilon > tolerance:
            Ta_V = np.matmul(self.T, V)
            all_values = self.R + self.discount * Ta_V
            V_new = np.max(all_values, axis=0)

            epsilon = np.linalg.norm(V_new - V, np.inf)
            V = V_new
            iterId += 1

        return [V, iterId, epsilon]

    def extractPolicy(self, V):
        all_values = self.R + self.discount * np.matmul(self.T, V)
        return np.argmax(all_values, axis=0)

    def evaluatePolicy(self, policy):
        R_policy = np.array([self.R[policy[i]][i] for i in range(len(policy))])
        T_policy = np.array([self.T[policy[i]][i] for i in range(len(policy))])
        V = np.matmul(
            np.linalg.inv(np.identity(len(policy)) - self.discount * T_policy),
            R_policy
        )
        return V

    def policyIteration(self, initialPolicy, nIterations=np.inf):
        policy = initialPolicy
        V = np.zeros(self.nStates)
        iterId = 0

        while iterId < nIterations:
            V = self.evaluatePolicy(policy)
            new_policy = self.extractPolicy(V)

            if np.array_equal(new_policy, policy):
                break

            policy = new_policy
            iterId += 1

        return [policy, V, iterId]