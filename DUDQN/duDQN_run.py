import collections
import gymnasium as gym
import random
import torch
import torch.nn.functional as F
import rl_utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from duDQN_brain import DQN ,ReplayBuffer

lr=2e-3
num_episodes=500
hidden_dim=128
gamma=0.95
max_epsilon=0.2
min_epsilon=0.01
decay=0.1

target_frequence=10
buffer_size=10000
minimal_size=300
batch_size=64
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')


env=gym.make('CartPole-v0')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer=ReplayBuffer(buffer_size)
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n




agent=DQN(state_dim,hidden_dim,action_dim,lr,gamma,max_epsilon,min_epsilon,decay,target_frequence,device)
return_list=[]
ic=0
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return=0
            state=env.reset()[0]
            done=0
            ic+=1
            while not done:
                action=agent.take_action(state,ic)
                next_state,reward,done,_,_=env.step(action)
                replay_buffer.add(state,action,reward,next_state,done)
                state=next_state
                episode_return+=reward
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
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })

            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format('CartPole-v0'))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format('CartPole-v0'))
plt.show()