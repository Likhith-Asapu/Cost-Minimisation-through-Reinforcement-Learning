import torch  
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Constants
X_0 = np.array([[5], 
                [-1]])
A = np.array([[0.9974, 0.0539], 
              [-0.1078, 1.1591]])
B = np.array([[0.0013], 
              [0.0539]])
Q = np.array([[0.25, 0], 
              [0, 0.5]])
R = 0.15
N = 121


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, learning_rate=1e-2):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.tanh(self.linear1(state))
        x = self.linear2(x)
        return x 
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        distribution = self.forward(Variable(state))
        mu = distribution.squeeze(0)[0]
        sig = 1
        action = torch.tensor([torch.normal(mu, sig)])
        probability = torch.exp(-0.5 * (((action - mu)/sig) ** 2)) / (torch.sqrt(torch.tensor([2 * np.pi])) * sig)
        log_prob = torch.log(probability)
        return action.item(), log_prob
    

def update_policy(policy_network, rewards, log_probs):
    
    discounted_rewards = []
    Gt = 0
    for t in range(len(rewards) - 1, -1, -1):
        Gt += rewards[t]
        discounted_rewards.append(Gt)
    discounted_rewards = discounted_rewards[::-1]
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
    
    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


def next_output(X,U):
    X = X.reshape(X.shape[0],1)
    reward = ((X.T @ Q @ X) + (R * U * U))/2
    return (A @ X + B * U).reshape(X.shape[0]),reward


def main():
    policy_net = PolicyNetwork(X_0.shape[0], 128)
    
    max_episode_num = 10000
    max_steps = 120
    numsteps = []
    Us = []
    X0s = []
    X1s = []
    all_rewards = []
    J = []
    for episode in range(max_episode_num):
        log_probs = []
        rewards = []
        state = X_0.reshape(X_0.shape[0])
        for steps in range(max_steps):
            action, log_prob = policy_net.get_action(state)
            new_state, reward = next_output(state,action)
            log_probs.append(log_prob)
            rewards.append(reward)
            if episode == max_episode_num - 1:
                Us.append(action)
                X0s.append(state[0])
                X1s.append(state[1])
                J.append(reward[0][0])
            state = new_state
        update_policy(policy_net, rewards, log_probs)
        all_rewards.append(np.sum(rewards))
        numsteps.append(episode)
        if episode % 1 == 0:
            print("episode: {}, total reward: {}".format(episode, np.sum(rewards)))
        
            
    
    J_rev = []
    Gt = 0
    for t in range(len(J) - 1, -1, -1):
        Gt += J[t]
        J_rev.append(Gt)
    J = J_rev[::-1]

X0s, X1s, J, Us, all_rewards = main()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
fig.subplots_adjust(wspace=0.2)  # Set horizontal space
fig.subplots_adjust(hspace=0.5)  # Set vertical space
Ns = np.arange(1,N)
axes[0][0].plot(Ns, X0s, lw=4, color='red')
axes[0][0].set_xlabel("Number of steps")
axes[0][0].set_ylabel(r"State ($x_0$)")

axes[0][1].plot(Ns, X1s, lw=4, color='orange')
axes[0][1].set_xlabel("Number of steps")
axes[0][1].set_ylabel(r"State ($x_1$)")

axes[1][0].plot(Ns[0:N-1], Us, lw=4, color='blue')
axes[1][0].set_xlabel("Number of steps")
axes[1][0].set_ylabel("Control Input (U)")

axes[1][1].plot(Ns, J, lw=4, color='green')
axes[1][1].set_xlabel("Number of steps")
axes[1][1].set_ylabel("Cost (J)")

plt.show()