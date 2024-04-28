import random
import torch
import torch.nn.functional as F
import rl_utils
from tqdm import tqdm
import numpy as np
import collections


class ReplayBuffer:
    '用于经验回访池'
    def __init__(self,capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    def sample(self,batch_size):
        transitions=random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done = zip(*transitions)
        return np.array(state),np.array(action),reward,np.array(next_state),done
    def size(self):
        return len(self.buffer)
class Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q

class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learn_rate,gamma,
                 max_epsilon,min_epsilon,decay,target_update,device):
        self.action_dim=action_dim
        self.state_dim=state_dim
        self.hidden_dim=hidden_dim
        self.learn_rate=learn_rate
        self.gamma=gamma
        self.qnet=Net(self.state_dim,self.hidden_dim,self.action_dim)
        self.target_qnet=Net(self.state_dim,self.hidden_dim,self.action_dim)
        self.epsilon=max_epsilon
        self.max_epsilon=max_epsilon
        self.min_epsilon=min_epsilon
        self.decay=decay
        self.target_frequece=target_update
        self.device=device
        self.count=0
        self.optimizer=torch.optim.Adam(self.qnet.parameters(),lr=self.learn_rate)
    def take_action(self,state,ic):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * ic)
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.action_dim)
        else:
            state=torch.tensor([state],dtype=torch.float).to(self.device)
            action=self.qnet(state).argmax().item()
        return action
    def learn(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions=torch.tensor(transition_dict['actions'],dtype=torch.long).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        next_states=torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
        q_values=self.qnet(states).gather(1,actions)
        max_next_q_values = self.target_qnet(next_states).max(dim=1, keepdim=True)[0]
        q_targets=rewards+self.gamma*max_next_q_values*(1-dones)
        loss=torch.mean(F.mse_loss(q_values,q_targets))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.count += 1
        if self.count%self.target_frequece==0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())




