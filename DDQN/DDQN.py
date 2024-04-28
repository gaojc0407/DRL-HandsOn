import random
import gymnasium as gym
import numpy as np
import collections
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from rl_utils import ReplayBuffer
from tqdm import tqdm
learn_rate = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = [0.2,0.01,0.01]
target_frequence = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 64
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')
env_name = "CartPole-v0"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n   # 将连续动作分成11个离散动作


class Net(torch.nn.Module):
    def __init__ (self,state_dim,hidden_dim,action_dim):
        super(Net,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,action_dim)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        return x

class DDQN:
    def __init__(self,state_dim,action_dim,hidden_dim,learn_rate,gamma,epsilon,target_frequence,device):
        self.action_dim=action_dim
        self.state_dim=state_dim
        self.hidden_dim=hidden_dim
        self.epsilon=epsilon
        self.gamma=gamma
        self.count=0
        self.device=device
        self.frequence=target_frequence
        self.qnet=Net(state_dim,hidden_dim,action_dim).to(device)
        self.tqnet=Net(state_dim,hidden_dim,action_dim).to(device)
        self.optimizer=torch.optim.Adam(self.qnet.parameters(),lr=learn_rate)
    def take_action(self,state,eposides):
        epsilon = self.epsilon[1] + (self.epsilon[0] - self.epsilon[1]) * np.exp(-self.epsilon[2] * eposides)
        if np.random.random()<epsilon:
            action=np.random.randint(self.action_dim)
        else:
            state = np.array([state], dtype=np.float32)  # 转换为浮点数并增加一个维度
            state = torch.tensor(state).to(device)
            action=self.qnet(state).argmax().item()
        return action

    def maxq(self,state):
        state=torch.tensor(state,dtype=torch.float).to(self.device)
        maxq=self.qnet(state).max().item()
        return maxq

    def learn(self,transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        qval=self.qnet(states).gather(1,actions)
        max_action=self.qnet(next_states).max(1)[1].view(-1,1)
        max_next_q=self.tqnet(next_states).gather(1,max_action)
        qtargets=rewards+self.gamma*max_next_q*(1-dones)
        loss=torch.mean(F.mse_loss(qval,qtargets))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.count%self.frequence==0:
            self.tqnet.load_state_dict(self.qnet.state_dict())
        self.count+=1

return_list=[]
max_qval_list=[]
max_qval=0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)

agent=DDQN(state_dim,action_dim,hidden_dim,learn_rate,gamma,epsilon,target_frequence,device)
ic=0
for i in range(10):
    with tqdm(total=int(num_episodes / 10),desc='Iteration %d' % i) as pbar:
        for i_e in range(int(num_episodes/10)):

            eposide_return=0
            state=env.reset()[0]
            done=0
            ic+=1
            while not done:
                action=agent.take_action(state,ic)
                max_qval=agent.maxq(state)*0.05+max_qval*0.95
                max_qval_list.append(max_qval)

                next_state,reward,done,_,_=env.step(action)
                replay_buffer.add(state,action,reward,next_state,done)
                state=next_state
                eposide_return+=reward
                if replay_buffer.size()>minimal_size:
                    s,a,r,ns,d=replay_buffer.sample(batch_size)
                    transition_dict={
                        'states':s,
                        'actions':a,
                        'next_states':ns,
                        'rewards':r,
                        'dones':d
                    }
                    agent.learn(transition_dict)
            return_list.append(eposide_return)
            if (i_e + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_e + 1),
                    'return':
                        '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)





episodes_list = list(range(len(return_list)))
mv_return = rl_utils.moving_average(return_list, 5)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Double DQN on {}'.format(env_name))
plt.show()

frames_list = list(range(len(max_qval_list)))
plt.plot(frames_list, max_qval_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('Double DQN on {}'.format(env_name))
plt.show()























