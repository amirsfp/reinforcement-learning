import numpy as np


class MultiArmedBandit:

    def __init__(self, k_arm=10, eps=0.1, initial=0., 
    UCB_constant=0.1, sample_averages_flag=False, step_size=0.1):
        self.k = k_arm
        self.q_true = np.random.uniform(low=-3.0, high=3.0, size=(self.k,))
        self.q_star = np.max(self.q_true)
        self.optimal_action_count = 0
        self.indices = np.arange(self.k)
        self.eps = eps
        self.initial = initial
        self.q_estimation = np.zeros((self.k,)) + self.initial
        self.action_count = np.zeros((self.k,))
        self.c = UCB_constant
        self.t = 0
        self.sample_averages_flag = sample_averages_flag
        self.step_size = step_size
    
    def reset(self):

        self.t = 0
        self.q_true = np.random.uniform(low=-3.0, high=3.0, size=(self.k,))
        self.q_star = np.max(self.q_true)
        self.optimal_action_count = 0
        self.q_estimation = np.zeros((self.k,)) + self.initial
        self.action_count = np.zeros((self.k,))

    def act(self):

        if np.random.rand() < self.eps:
            action = np.random.choice(self.indices)
        
        else:
            upper_confidence_bound = self.q_estimation + \
                self.c * np.sqrt( np.log(self.t + 1) / (self.action_count + 1e-5) )
            q_best = np.max(upper_confidence_bound)
            action = np.random.choice(np.where(upper_confidence_bound == q_best)[0])
        
        if self.q_true[action] == self.q_star:
            self.optimal_action_count += 1
        
        return action
    
    def step(self, action):

        self.t += 1
        reward = np.random.normal(loc=self.q_true[action], scale=1)
        self.action_count[action] += 1

        if self.sample_averages_flag:
            self.q_estimation[action] = (reward - self.q_estimation[action]) \
                / self.action_count[action]
        
        else:
            self.q_estimation[action] = self.step_size * (reward - self.q_estimation[action])
        
        return reward
    
    def get_optimal_action_ratio(self):
        if self.optimal_action_count:
            return self.optimal_action_count / self.t
        else:
            return None


if __name__ == "__main__":

    RUNS = 2000
    TIME = 1000
    eps = [0.1, 0.01, 0.001]
    bandits = [MultiArmedBandit(eps=eps) for eps in eps]
    rewards = np.zeros((len(bandits), RUNS, TIME))
    best_action_ratios = np.zeros(rewards.shape)

    for b, bandit in enumerate(bandits):
        for run in range(RUNS):
            for t in range(TIME):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[b, run, t] = reward
                best_action_ratios[b, run, t] = bandit.get_optimal_action_ratio()

            bandit.reset()
            if run % 100 == 0 and run != 0:
                print(run)
        print(b)
        
    mean_rewards = rewards.mean(axis=1)
    mean_best_action_ratios = best_action_ratios.mean(axis=1)

