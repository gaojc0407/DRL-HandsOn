import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import collections

class PolicyNet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(PolicyNet,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.softmax(x,dim=1)
        return x

class QvalueNet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(QvalueNet,self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        return x


class SAC:
    def __init__(self,state_dim,hidden_dim,action_dim,actor_lr,critic_lr,alpha_lr,target_entropy,tau,gamma,device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QvalueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QvalueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QvalueNet(state_dim, hidden_dim, action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QvalueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_1_optimizer=torch.optim.Adam(self.critic_1.parameters(),lr=critic_lr)
        self.critic_2_optimizer=torch.optim.Adam(self.critic_2.parameters(),lr=critic_lr)
        self.log_alpha=torch.tensor(np.log(1e-2),dtype=torch.float)
        self.log_alpha.requires_grad=True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=alpha_lr)
        self.gamma=gamma
        self.tau=tau
        self.t_entropy=target_entropy
        self.device=device
    def take_action(self,state):
        state=torch.tensor([state],dtype=torch.float).to(self.device)
        probs=self.actor(state)
        action_dist=torch.distributions.Categorical(probs)
        action=action_dist.sample()
        return action.item()

    def soft_update(self,net,t_net):
        for param,t_param in zip(net.parameters(),t_net.parameters()):
            t_param.data.copy_(t_param.data*(1-self.tau)+param.data*self.tau)



    def learn(self,transition_dict):
       states=torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
       actions=torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
       rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
       next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
       dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)

       next_probs=self.actor(next_states)
       next_log_probs=torch.log(next_probs+0.0000000001)
       entropy=-torch.sum(next_probs*next_log_probs,dim=1,keepdim=True)
       q1value=self.target_critic_1(next_states)
       q2value=self.target_critic_2(next_states)
       need_q=torch.sum(next_probs*torch.min(q1value,q2value),dim=1,keepdim=True)
       next_value=need_q+self.log_alpha.exp()*entropy
       td_target=rewards+self.gamma*next_value*(1-dones)

       critic_1_qval=self.critic_1(states).gather(1,actions)
       critic_2_qval=self.critic_2(states).gather(1,actions)
       critic_1_loss=torch.mean(F.mse_loss(critic_1_qval,td_target.detach()))
       critic_2_loss =torch.mean(F.mse_loss(critic_2_qval, td_target.detach()))
       self.critic_1_optimizer.zero_grad()
       self.critic_2_optimizer.zero_grad()
       critic_1_loss.backward()
       critic_2_loss.backward()
       self.critic_1_optimizer.step()
       self.critic_2_optimizer.step()

       probs=self.actor(states)
       log_probs=torch.log(probs+0.000001)
       entropy_a=-torch.sum(probs*log_probs,dim=1,keepdim=True)
       now_q1val=self.critic_1(states)
       now_q2val=self.critic_2(states)
       min_qvalue=torch.sum(probs*torch.min(now_q1val,now_q2val),dim=1,keepdim=True)
       actor_loss=torch.mean(-self.log_alpha.exp()*entropy_a-min_qvalue)
       self.actor_optimizer.zero_grad()
       actor_loss.backward()
       self.actor_optimizer.step()

       alpha_loss=torch.mean((entropy_a-self.t_entropy).detach()*self.log_alpha.exp())
       self.log_alpha_optimizer.zero_grad()
       alpha_loss.backward()
       self.log_alpha_optimizer.step()

       self.soft_update(self.critic_1, self.target_critic_1)
       self.soft_update(self.critic_2, self.target_critic_2)


actor_lr = 1e-3
critic_lr = 1e-2
alpha_lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 500
batch_size = 64
target_entropy = -1
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')

env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
            target_entropy, tau, gamma, device)

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=collections.deque(maxlen=capacity)
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    def sample(self,batch_size):
        transitions=random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done

    def size(self):
        return len(self.buffer)

replay_buffer=ReplayBuffer(buffer_size)
return_list=[]
for i in range(10):
    with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
        for i_e in range(int(num_episodes/10)):
            episode_return=0
            state=env.reset()[0]
            done=0
            while not done:
                action=agent.take_action(state)
                next_state,reward,done,_,_=env.step(action)
                replay_buffer.add(state,action,reward,next_state,done)
                state=next_state
                episode_return+=reward
                if replay_buffer.size()>minimal_size:
                    s,a,r,ns,d=replay_buffer.sample(batch_size)
                    tranisitin_dict={'states':s,'actions':a,'next_states':ns,'rewards':r,'dones':d}
                    agent.learn(tranisitin_dict)
            return_list.append(episode_return)
            if(i_e+1)%10==0:
                pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_e+1), 'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

episodes_list=list(range(len(return_list)))
plt.plot(episodes_list,return_list)
plt.xlabel('episodes')
plt.ylabel('returns')
plt.title('SAC on {}'.format(env_name))
plt.show()



























