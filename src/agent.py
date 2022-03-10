import copy
# import random
import torch
import utils
import numpy as np
from model import Network


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions) -> None:
        """Class to store agent's experience"""
        self.max_size = max_size
        self.memory_count = 0

        # Experience = (s, a, r, s', d)
        self.states = np.zeros((self.max_size, *input_shape), dtype=np.uint8)
        self.actions = np.zeros((self.max_size),dtype=np.int8)
        self.rewards = np.zeros(self.max_size)
        self.next_states = np.zeros((self.max_size, *input_shape), dtype=np.uint8)
        self.done = np.zeros(self.max_size, dtype=np.float32)
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add an experience to memory"""
        index = self.memory_count % self.max_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.done[index] = 1-done
        self.memory_count += 1

    def recall(self, batch_size):
        """Sample experiences from memory"""
        memory_len = np.min([self.memory_count, self.max_size])
        batch = np.random.choice(memory_len, batch_size, replace=False)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        done = self.done[batch]

        return states, actions, rewards, next_states, done

class Agent(object):
    def __init__(self, alpha, gamma, epsilon, epsilon_decay, max_size, burnin, input_shape, \
            n_actions, batch_size, save_every, update_every, train_every, save_dir, checkpoint) -> None:
        
        self.burnin = burnin
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.exploration_decay_rate = epsilon_decay
        self.curr_step = 0
        self.update_every = update_every
        self.gamma = gamma
        self.train_online = train_every
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.input_shape = input_shape
        self.save_every = save_every
        self.temp = 0
        self.temp_previous = None
        
        self.replay_memory = ReplayBuffer(max_size, input_shape, self.n_actions)
        self.online = Network(alpha=alpha, input_dim=input_shape, output_dim=self.n_actions, checkpoint=save_dir)
        self.target = copy.deepcopy(self.online)
        # Network parameters
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.target.freeze_paramaters()

        if checkpoint!=None:
            self.load_model()
        

    def act(self, state):
        if np.random.rand()<self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            # Forward pass on state to get Q-values
            self.online.eval()
            state = state.reshape((1, *self.input_shape))
            state = state.__array__()
            state = torch.tensor(state, dtype=torch.float).to(self.online.device)
            action_idx = np.argmax(self.online.forward(state).cpu().detach().numpy())
            self.online.train()

        self.epsilon *= self.exploration_decay_rate
        self.epsilon = max(self.epsilon, self.epsilon_min)
        self.curr_step += 1
        return action_idx

    def cache(self, state, action, reward, next_state, done):
        self.replay_memory.add_experience(state, action, reward, next_state, done)

    def learn(self):
        
        # Update target network
        if self.curr_step % self.update_every == 0:
            # if self.temp==1:
            #     temp = copy.deepcopy(self.target.state_dict())

            self.target.load_state_dict(self.online.state_dict())
            # if self.temp == 1:
            #     print('Target params remain frozen: ', utils.validate_state_dicts(temp, self.temp_previous))
            #     print('Target params updated: ', True if not utils.validate_state_dicts(self.temp_previous, self.target.state_dict()) else False)
            #     self.temp = 0
            # if self.temp==0:
            #     self.temp_previous = copy.deepcopy(self.target.state_dict())
            #     self.temp = 1

        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step % self.save_every == 0:
            self.save_model()
        
        if not self.curr_step % self.train_online == 0:
            return None, None
        
        # Get minibatch
        states, actions, rewards, next_states, done = self.replay_memory.recall(self.batch_size)

        states = torch.tensor(states, dtype=torch.float).to(self.online.device)
        actions = torch.tensor(actions, dtype=torch.int8).to(self.online.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.online.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.online.device)
        done = torch.tensor(done).to(self.online.device)

        self.target.eval()
        target_qs = self.target.forward(next_states)

        target = []
        for j in range(self.batch_size):
            target.append(rewards[j] + self.gamma*torch.max(target_qs[j])*(1-done[j]))
        self.target.train()
        target = torch.tensor(target).to(self.online.device)
        target = target.view(self.batch_size)

        # Forward pass
        self.online.eval()
        online_qs = self.online.forward(states)
        online_qs = online_qs[np.arange(0, self.batch_size), actions.long()]  # Q_online(s,a)

        # Gradient Descent on online network
        self.online.train()
        self.online.optimizer.zero_grad()
        online_loss = self.loss_fn(online_qs, target)
        online_loss.backward()
        self.online.optimizer.step()
        self.online.eval()

        return online_qs.mean().item(), online_loss.cpu().detach().numpy() 

    def save_model(self):
        self.online.save_checkpoint(self.epsilon, self.curr_step//self.save_every)

    def load_model(self):
        self.epsilon = self.online.load_checkpoint(self.checkpoint)
        self.target.load_state_dict(self.online.state_dict())
        # print('exploration rate: ', self.epsilon)

if __name__=="__main__":
    pass
