import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst = [], [], [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(self):
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.fc2 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, last_action, hidden_in):
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)

        h1 = torch.cat([state, action], -1)
        h1 = F.relu(self.fc1(h1))

        h2 = torch.cat([state, last_action], -1)
        h2 = F.relu(self.fc2(h2))
        h2, hidden_out = self.lstm1(h2, hidden_in)

        h = torch.cat([h1, h2], -1)
        x = F.relu(self.fc3(h))
        x = self.fc4(x)
        x = x.permute(1, 0, 2)  # back to same axes as input
        return x, hidden_out

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = -20
        self.log_std_max = 2
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.state_dim, hidden_dim)
        self.fc2 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, self.action_dim)
        self.log_std = nn.Linear(hidden_dim, self.action_dim)

    def sample_action(self, ):
        a = torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return self.action_range * a.numpy()

    def forward(self, state, last_action, hidden_in):
        state = state.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)

        h1 = F.relu(self.fc1(state))

        h2 = torch.cat([state, last_action], -1)
        h2 = F.relu(self.fc2(h2))
        h2, hidden = self.lstm1(h2, hidden_in)

        h = torch.cat([h1, h2], -1)
        x = F.relu(self.fc3(h))
        x = F.relu(self.fc4(x))
        x = x.permute(1, 0, 2)  # permute back

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std, hidden

    def evaluate(self, state, last_action, hidden_in):
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp()
        z = Normal(0, 1).sample(mean.shape)
        action_0 = torch.tanh(mean + std * z)
        action = action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1. - action_0.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std, hidden_out

    def get_action(self, state, last_action, hidden_in):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0)
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp()
        z = Normal(0, 1).sample(mean.shape)
        action = torch.tanh(mean + std * z)
        action = action.detach().numpy()
        return action[0][0], hidden_out

class SAC_Trainer():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = Critic(state_dim, action_dim, hidden_dim)
        self.soft_q_net2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net1 = Critic(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net2 = Critic(state_dim, action_dim, hidden_dim)
        self.policy_net = Actor(state_dim, action_dim, hidden_dim)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        self.gamma = 0.99
        self.soft_tau = 1e-2
        self.reward_scale = 10.0
        self.target_entropy = -1. * action_dim
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=3e-4)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def update(self, batch_size):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        last_action = torch.FloatTensor(last_action)
        reward = torch.FloatTensor(reward).unsqueeze(-1)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1)

        predicted_q_value1, _ = self.soft_q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, last_action, hidden_in)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
        reward = self.reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

        # Updating alpha wrt entropy
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Training Q Function
        predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, hidden_out)
        predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * self.gamma * target_q_min
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predict_q1, _ = self.soft_q_net1(state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '_q1')
        torch.save(self.soft_q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path + '_q2'))
        self.policy_net.load_state_dict(torch.load(path + '_policy'))
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

if __name__ == '__main__':
    replay_buffer = ReplayBuffer(capacity=1e6)

    # choose env
    env = gym.make('Pendulum-v0')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    max_episodes = 100
    max_steps = 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
    batch_size = 2
    hidden_dim = 256
    rewards = []
    model_path = './model/sac_v2_lstm'

    sac_trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, hidden_dim=hidden_dim)
    for eps in range(max_episodes):
        state = env.reset()
        last_action = env.action_space.sample()
        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []
        hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float), torch.zeros([1, 1, hidden_dim]))
        for step in range(max_steps):
            hidden_in = hidden_out
            action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in)
            action = action*2
            next_state, reward, done, _ = env.step(action)
            # env.render()

            if step == 0:
                ini_hidden_in = hidden_in
                ini_hidden_out = hidden_out

            # state: np.array([state_dim, ])
            # action: np.array([action_dim, ])
            # reward: float
            # done: boolean
            episode_state.append(state)
            episode_action.append(action)
            episode_last_action.append(last_action)
            episode_reward.append(reward)
            episode_next_state.append(next_state)
            episode_done.append(done)

            state = next_state
            last_action = action

            if len(replay_buffer) > batch_size:
                sac_trainer.update(batch_size)
            if done:
                break
        replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, episode_reward, episode_next_state, episode_done)

        print('Episode: ', eps, '| Episode Reward: ', np.sum(episode_reward))
        rewards.append(np.sum(episode_reward))
