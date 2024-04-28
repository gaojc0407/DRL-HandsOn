import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)




class PolicyNet(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim):
        super(PolicyNet,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,action_dim)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        x=self.fc3(x)
        x=F.softmax(x,dim=1)
        return x

class ValueNet(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim):
        super(ValueNet,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,action_dim)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        x=self.fc3(x)
        return x

class PPO:
    def __init__(self,state_dim,hidden_dim,action_dim,alr,clr,lamda,epochs,eps,gamma,device):
        self.actor=PolicyNet(state_dim,action_dim,hidden_dim).to(device)
        self.critic=ValueNet(state_dim,action_dim,hidden_dim).to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),alr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),clr)
        self.gamma=gamma
        self.lamda=lamda
        self.epochs=epochs
        self.eps=eps
        self.device=device

    def take_action(self,state):
        state=torch.tensor([state],dtype=torch.float).to(self.device)
        probs=self.actor(state)
        action_dist=torch.distributions.Categorical(probs)
        action=action_dist.sample()
        return action.item()

    def learn(self,transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        td_target=rewards+self.gamma*self.critic(next_states)*(1-dones)
        td_delta=td_target-self.critic(states)
        advantage=compute_advantage(self.gamma,self.lamda,td_delta.cpu()).to(self.device)
        old_log_probs=torch.log(self.actor(states).gather(1,actions)).detach()
        for i in range(self.epochs):
            log_probs=torch.log(self.actor(states).gather(1,actions))
            ratio=torch.exp(log_probs-old_log_probs)
            surr1=ratio*advantage
            surr2=torch.clamp(ratio,1-self.eps,1+self.eps)*advantage
            actor_loss=torch.mean(-torch.min(surr1,surr2))
            critic_loss=torch.mean(F.mse_loss(self.critic(states),td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()





alr=0.003
clr=0.01
episodes=500
hidden_dim=128
gamma=0.97
lamda=0.95
epochs=10
eps=0.2
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')

env_name="CartPole-v1"
env=gym.make(env_name)
torch.manual_seed(0)
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n

agent=PPO(state_dim,hidden_dim,action_dim,alr,clr,lamda,epochs,eps,gamma,device)

return_list = []
for i in range(10):
    with tqdm(total=int(episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(episodes / 10)):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()[0]
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done,_,_ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.learn(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 21)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()