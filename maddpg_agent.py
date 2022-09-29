# This code is based on the ddpg-pendulum github of udacity nanodegree deep reinforcement learning.

import numpy as np
import random
import copy
from collections import namedtuple, deque

import model as m

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1        # how often to update the network
NUM_STEP = 1            # how many gradient ascent step


expand=lambda x:np.expand_dims(x,axis=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG():
    
    def __init__(self, state_sizes, action_sizes, n_agents=None, random_seed=0, path="", discrete=False):
        """Initialize an Multi Agent object.
        
        Params
        ======
            state_size (list or int): dimension of each state and agent
            action_size (list or int): dimension of each action and agent
            random_seed (list or int): random seed of each agent
            n_agents (int or None): number of agents
            path (str): path to the checkpoint folder
            discrete  (bool): True if the action space is discrete else False
        """
        if n_agents is None:
            
            if type(state_sizes)==list:
                n_agents=len(state_sizes)
            
            elif type(action_sizes)==list:
                n_agents=len(action_sizes)
                
            else:
                
                raise ValueError
                
        if type(state_sizes)==int:
            state_sizes=[state_sizes,]*n_agents
                
        if type(action_sizes)==int:
            action_sizes=[action_sizes,]*n_agents
                
        assert len(action_sizes)==len(state_sizes)==n_agents
            
        self.n_agents=n_agents
        
        self.path=path + "/" if len(path)>0 else ""
        
        input_size=sum(state_sizes)+(n_agents if discrete else sum(action_sizes))
        
        self.memory = ReplayBuffer(n_agents,BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        self.agents=[MAgent(i, input_size, state_sizes[i], action_sizes[i], random_seed, discrete) for i in range(n_agents)]
        
        
        
        
    def reset(self):
        for i in range(self.n_agents):
            self.agents[i].noise.reset()
            
    def step(self, state, action, reward, next_state, done, t):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # expand first dimension to be gathered by batch
        
        state=list(map(expand,state))
        reward=list(map(expand,reward))
        next_state=list(map(expand,next_state))
        done=list(map(expand,done))
        
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and t% UPDATE_EVERY==0:
            
            
            states, actions, rewards, next_states, dones = self.memory.sample()
            
            actions_next=self.actions_next(next_states)
            
            state_action=torch.cat(states+actions ,dim=1)
            
            next_state_action=torch.cat(next_states+actions_next ,dim=1)
            
            for i in range(self.n_agents):
                for _ in range(NUM_STEP):
                    
                    experiences=[states, actions, state_action, rewards[i], dones[i], next_state_action]
                    
                    self.agents[i].learn(experiences, GAMMA)
                    
                    
    
    def act(self, states, param):
        """Returns actions for given state as per current policy for each agent."""
        agents_action=list()
        states=list(map(expand,states))
        for i in range(self.n_agents):
            agents_action.append(self.agents[i].act(states[i], param))
            
        return agents_action
    
    def actions_next(self,states):
        "Get predicted next-state actions"
        
        agents_action=list()
        with torch.no_grad():
            for i in range(self.n_agents):
                agents_action.append(self.agents[i].actor_target(states[i]))
            
        return agents_action
    
    
    def save(self,idx):
        "Save each netorwk for each agent"
        for i in idx:
            torch.save(self.agents[i].actor_local.state_dict(), self.path+'checkpoint_actor '+str(i)+'.pth')
            torch.save(self.agents[i].critic_local.state_dict(), self.path+'checkpoint_critic '+str(i)+'.pth')
            
    def load(self):
        "Load local netorwks for each agent"
        for i in range(self.n_agents):
            self.agents[i].actor_local.load_state_dict(torch.load(self.path+'checkpoint_actor '+str(i)+'.pth'))
            self.agents[i].critic_local.load_state_dict(torch.load(self.path+'checkpoint_critic '+str(i)+'.pth'))


class MAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, idx, input_size, state_size, action_size, random_seed, discrete):
        """Initialize an Agent object.
        
        Params
        ======
            idx (int): id of the agent
            input_size (int): sum of states and actions size
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            discrete  (bool): True if the action space is discrete else False
        """
        
        self.idx=idx
        
        self.seed = random.seed(random_seed)
        
        self.action_size = action_size

        # Actor Network (w/ Target Network)
        
        if discrete:
            
            self.actor_local = m.ActorD(state_size, action_size, random_seed).to(device)
            self.actor_target = m.ActorD(state_size, action_size, random_seed).to(device)
            
        else:
            
            self.actor_local = m.ActorC(state_size, action_size, random_seed).to(device)
            self.actor_target = m.ActorC(state_size, action_size, random_seed).to(device)
            # Noise process
            self.noise = OUNoise(action_size, random_seed)
        
        
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        
        self.critic_local = m.MACritic(input_size, random_seed).to(device)
        self.critic_target = m.MACritic(input_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        
        


    def act(self, state, param):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        
        if type(param)==bool:
            if param:
                action += self.noise.sample()
            return np.clip(action, -1, 1)
        
        
        elif type(param)==float:
            # Epsilon-greedy action selection
            if np.random.random() > param:
                return action
            else:
                return np.random.choice(np.arange(self.action_size))


    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            i (int): current agent
        """
        states, actions, state_action, rewards, dones , next_state_action = experiences
        
        # ---------------------------- update critic ---------------------------- #
        
        with torch.no_grad():
            # Get Q values from target models
            Q_targets_next = self.critic_target(next_state_action)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next* (1 - dones))
            
        # Compute critic loss
        Q_expected = self.critic_local(state_action)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        
        action = self.actor_local(states[self.idx])
        
        actions=actions[:self.idx]+[action,]+actions[self.idx+1:]
        
        state_action=torch.cat(states+actions ,dim=1)
        
        actor_loss = -self.critic_local(state_action).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.size=size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self,n_agents, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.n_agents=n_agents
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        for i in range(self.n_agents):
            
            
            for j in range(BATCH_SIZE):
                
                e=experiences[j]
                
                if e.state[i] is not None:
                    if j==0:
                        state=e.state[i]
                    else :
                        state=np.vstack((state,e.state[i]))
                    
                if e.action[i] is not None:
                    if j==0:
                        action=e.action[i]
                    else :
                        action=np.vstack((action,e.action[i]))
                
                if e.reward[i] is not None:
                    if j==0:
                        reward=e.reward[i]
                    else :
                        reward=np.vstack((reward,e.reward[i]))
                 
                if e.next_state[i] is not None:
                    if j==0:
                        next_state=e.next_state[i]
                    else :
                        next_state=np.vstack((next_state,e.next_state[i]))
                    
                if e.done[i] is not None:
                    if j==0:
                        done=e.done[i]
                    else :
                        done=np.vstack((done,e.done[i]))
                        
            
            state=torch.from_numpy(state).float().to(device)
            action=torch.from_numpy(action).float().to(device)
            reward=torch.from_numpy(reward).float().to(device)
            next_state=torch.from_numpy(next_state).float().to(device)
            done=torch.from_numpy(done.astype(np.uint8)).float().to(device)
            
            if i==0:
                
                states=[state,]+[torch.zeros_like(state),]*(self.n_agents-1)
                actions=[action,]+[torch.zeros_like(action),]*(self.n_agents-1)
                rewards=[reward,]+[torch.zeros_like(reward),]*(self.n_agents-1)
                next_states=[next_state,]+[torch.zeros_like(next_state),]*(self.n_agents-1)
                dones=[done,]+[torch.zeros_like(done),]*(self.n_agents-1)
                
            else:
                
                states[i]=state
                actions[i]=action
                rewards[i]=reward
                next_states[i]=next_state
                dones[i]=done

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)