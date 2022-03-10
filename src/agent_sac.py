#########################################################################
#   Implementation of Soft Actor Critic by Josias Moukpe
#   from the https://arxiv.org/abs/1812.05905 paper
#   and inspired by https://github.com/pranz24/pytorch-soft-actor-critic
##########################################################################

import copy

from numpy.lib.function_base import select
# import random
import torch
import numpy as np
from torch._C import dtype
from model_sac import ValueNetwork, QNetwork, PolicyNetwork
import os
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions) -> None:
        """Class to store agent's experience"""
        self.max_size = max_size
        self.memory_count = 0

        # Experience = (s, a, r, s', d)
        self.states = {
            'camera': np.zeros((self.max_size, *input_shape), dtype=np.uint8),
            'lidar': np.zeros((self.max_size, *input_shape), dtype=np.uint8),
            'birdeye': np.zeros((self.max_size, *input_shape), dtype=np.uint8)
        }
        
        self.actions = np.zeros((self.max_size, n_actions),dtype=np.float32)
        self.rewards = np.zeros(self.max_size)
        self.next_states = {
            'camera': np.zeros((self.max_size, *input_shape), dtype=np.uint8),
            'lidar': np.zeros((self.max_size, *input_shape), dtype=np.uint8),
            'birdeye': np.zeros((self.max_size, *input_shape), dtype=np.uint8)
        }#np.zeros((self.max_size, *input_shape), dtype=np.uint8)
        self.done = np.zeros(self.max_size, dtype=np.float32)
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add an experience to memory"""
        index = self.memory_count % self.max_size
        self.states[index] = state
        # print(action)
        # print('actions')
        # print(self.actions)
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
    '''Soft Actor Critic Agent'''
    def __init__(self, 
    alpha, 
    gamma, 
    tau, # tau = 1 by default
    epsilon, 
    epsilon_decay, 
    max_size, 
    burnin, 
    input_shape, 
    action_space, 
    batch_size, 
    save_every, 
    update_every, 
    train_every, 
    save_dir, 
    checkpoint) -> None:
        
        self.burnin = burnin
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.exploration_decay_rate = epsilon_decay
        self.curr_step = 0
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.train_online = train_every
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.input_shape = input_shape
        self.save_every = save_every
        self.temp = 0
        self.temp_previous = None
        self.replay_memory = ReplayBuffer(max_size, input_shape, action_space.shape[0])
        self.automatic_entropy_tuning = False
        self.target_update_interval = update_every

        # Defining the critic networks
        self.critic = QNetwork(
            alpha=alpha, 
            input_dim=input_shape, 
            action_dim=action_space.shape[0], 
            checkpoint=save_dir
        )
        
        self.critic_optim = self.critic.optimizer
        self.critic_target = QNetwork(
            alpha=alpha, 
            input_dim=input_shape, 
            action_dim=action_space.shape[0], 
            checkpoint=save_dir
        )

        hard_update(self.critic_target, self.critic)
        
        # defining the policy network
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=alpha)

        self.policy = PolicyNetwork(
            alpha=alpha, 
            input_dim=input_shape, 
            action_dim = action_space.shape[0], 
            action_space = action_space, 
            checkpoint=save_dir
        )
        self.policy_optim = self.policy.optimizer

        if checkpoint!=None:
            self.load_model()
        

    def act(self, state, evaluate=False):

        self.policy.eval()
        # state = torch.cat(
        #     torch.from_numpy(state['camera']), 
        #     torch.from_numpy(state['lidar']), 
        #     torch.from_numpy(state['birdeye'])
        # )
        state['camera'] = torch.tensor(state['camera'], dtype=torch.float)
        state['lidar'] = torch.tensor(state['lidar'], dtype=torch.float)
        state['birdeye'] = torch.tensor(state['birdeye'], dtype=torch.float)

        state['camera'] = state['camera'].reshape((1, *self.input_shape))
        state['lidar'] = state['lidar'].reshape((1, *self.input_shape))
        state['birdeye'] = state['birdeye'].reshape((1, *self.input_shape))

        # state = state.__array__()
        
        # state = torch.tensor(state, dtype=torch.float).to(self.policy.device)
        state['camera'] = state['camera'].to(self.policy.device)
        state['lidar'] = state['lidar'].to(self.policy.device)
        state['birdeye'] = state['birdeye'].to(self.policy.device)
        
        if evaluate is False:
            action, _, _ = self.policy.sample(state) # prefered
        else:
            _, _, action = self.policy.sample(state)

        return action.detach().cpu().numpy()[0]

    def learn(self, updates):
        '''Main training loop for soft actor critic'''
        # Update target network
        # if self.curr_step % self.update_every == 0:
        #     # if self.temp==1:
        #     #     temp = copy.deepcopy(self.target.state_dict())

        #     self.target.load_state_dict(self.online.state_dict())
        #     # if self.temp == 1:
        #     #     print('Target params remain frozen: ', utils.validate_state_dicts(temp, self.temp_previous))
        #     #     print('Target params updated: ', True if not utils.validate_state_dicts(self.temp_previous, self.target.state_dict()) else False)
        #     #     self.temp = 0
        #     # if self.temp==0:
        #     #     self.temp_previous = copy.deepcopy(self.target.state_dict())
        #     #     self.temp = 1

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

        # self.target.eval()
        # target_qs = self.target.forward(next_states)

        # target = []
        # for j in range(self.batch_size):
        #     target.append(rewards[j] + self.gamma*torch.max(target_qs[j])*(1-done[j]))
        # self.target.train()
        # target = torch.tensor(target).to(self.online.device)
        # target = target.view(self.batch_size)

        # # Forward pass
        # self.online.eval()
        # online_qs = self.online.forward(states)
        # online_qs = online_qs[np.arange(0, self.batch_size), actions.long()]  # Q_online(s,a)

        # # Gradient Descent on online network
        # self.online.train()
        # self.online.optimizer.zero_grad()
        # online_loss = self.loss_fn(online_qs, target)
        # online_loss.backward()
        # self.online.optimizer.step()
        # self.online.eval()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_states)
            qf1_next_target, qf2_next_target = self.critic_target(next_states, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + done * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(states, actions)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(states)
        qf1_pi, qf2_pi = self.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return policy_loss.item(), qf1_loss.item() + qf2_loss.item()
        
    def cache(self, state, action, reward, next_state, done):
        self.replay_memory.add_experience(state, action, reward, next_state, done)

    def save_model(self):
        self.policy.save_checkpoint(self.epsilon, self.curr_step//self.save_every)
        self.critic.save_checkpoint(self.epsilon, self.curr_step//self.save_every)
        self.critic_target.save_checkpoint(self.epsilon, self.curr_step//self.save_every)

    def load_model(self):
        self.epsilon = self.policy.load_checkpoint(self.checkpoint)
        self.target.load_state_dict(self.policy.state_dict())
        # print('exploration rate: ', self.epsilon)

     # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

if __name__=="__main__":
    pass
