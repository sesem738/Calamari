#########################################################################
#   Implementation of Soft Actor Critic by Josias Moukpe
#   from the https://arxiv.org/abs/1812.05905 paper
#   and inspired by https://github.com/pranz24/pytorch-soft-actor-critic
##########################################################################


# 12 channel : 3 lidar + 9 rgb


import torch
from torch import nn
from torch._C import device 
import torch.nn.functional as F
from torch.distributions import Normal 

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    '''initialize the po '''
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    '''Soft Actor Critic Value Network'''
    def __init__(self, alpha, input_dim, output_dim = 1, name='ValueNet', checkpoint='checkpoints/sac'):
        super(ValueNetwork, self).__init__()
        
        c, h, w = input_dim
        self.name = name
        self.checkpoint_file = checkpoint

        if h != 256:
            raise ValueError(f"Expecting input height: 150, got: {h}")
        if w != 256:
            raise ValueError(f"Expecting input width: 150, got: {w}")
        
        c1 = 3; c2 = 3; c3 = 3# 3 channels lidar, 9 channels rgb camera

        # Feature extraction
        # 1 -> camera
        self.conv11 = nn.Conv2d(in_channels=c1, out_channels=64, kernel_size=8, stride=4)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # 2 -> lidar
        self.conv21 = nn.Conv2d(in_channels=c2, out_channels=64, kernel_size=8, stride=4)
        self.conv22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv23 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # 3 -> birdeye
        self.conv31 = nn.Conv2d(in_channels=c3, out_channels=64, kernel_size=8, stride=4)
        self.conv32 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv33 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        out_size = 150528 #TODO: find out

        # Estimation 
        self.dense1 = nn.Linear(out_size, 512)
        self.dense2 = nn.Linear(512,256)
        self.dense3 = nn.Linear(256,output_dim) # output_dim is 1 for SAC Value function

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device('cuda:0')
        self.to(device=self.device)

        #Optimal initialization of the weights
        self.apply(weights_init_)

    def forward(self, input):
        '''Forward pass to the Value Network to estimate the value'''
        # input is the state 
        # input_1, _, input_2 = torch.tensor_split(input,(3,3), dim=1)

        # input_1 = F.relu(self.conv11(input_1))
        # input_1 = F.relu(self.conv12(input_1))
        # input_1 = F.relu(self.conv13(input_1))
        # input_1 = torch.flatten(input_1,1)

        # input_2 = F.relu(self.conv21(input_2))
        # input_2 = F.relu(self.conv22(input_2))
        # input_2 = F.relu(self.conv23(input_2))
        # input_2 = torch.flatten(input_2,1)

        # x = torch.cat((input_1, input_2),dim=-1)

        input_1 = input['camera']
        input_2 = input['lidar']
        input_3 = input['birdeye']

        input_1 = F.relu(self.conv11(input_1))
        input_1 = F.relu(self.conv12(input_1))
        input_1 = F.relu(self.conv13(input_1))
        input_1 = torch.flatten(input_1,1)

        input_2 = F.relu(self.conv21(input_2))
        input_2 = F.relu(self.conv22(input_2))
        input_2 = F.relu(self.conv23(input_2))
        input_2 = torch.flatten(input_2,1)

        input_3 = F.relu(self.conv31(input_3))
        input_3 = F.relu(self.conv32(input_3))
        input_3 = F.relu(self.conv33(input_3))
        input_3 = torch.flatten(input_3,1)

        x = torch.cat((input_1, input_2, input_3),dim=-1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x
    
    def freeze_paramaters(self):
        """Freeze network parameters"""
        for p in self.parameters():
            p.requires_grad = False
    
    
    # TODO: found out if it's affected by the update to SAC
    def save_checkpoint(self, epsilon, num):
        print('... saving checkpoint ...')
        path = self.checkpoint_file / (self.name+'_'+str(num)+'.chkpt')
        torch.save(dict(model=self.state_dict(), epsilon_decay=epsilon), path)
    
    def load_checkpoint(self, checkpoint_file):
        if not checkpoint_file.exists():
            raise ValueError(f"{checkpoint_file} does not exist")
        print('... loading checkpoint ...')
        ckp = torch.load(checkpoint_file)
        exploration_rate = ckp.get('epsilon_decay')
        state_dict = ckp.get('model')
        self.load_state_dict(state_dict)
        return exploration_rate

class QNetwork(nn.Module):
    '''Soft Actor Critic Q Network'''
    def __init__(self, alpha, input_dim, action_dim, output_dim =1, name='QNet', checkpoint='checkpoints/sac'):
        super(QNetwork, self).__init__()
        
        c, h, w = input_dim
        self.name = name
        self.checkpoint_file = checkpoint

        if h != 256:
            raise ValueError(f"Expecting input height: 150, got: {h}")
        if w != 256:
            raise ValueError(f"Expecting input width: 150, got: {w}")
        
        c1 = 3; c2 = 3; c3 = 3# 3 channels lidar, 9 channels rgb camera

        # Feature extraction
        # 1 -> camera
        self.conv11 = nn.Conv2d(in_channels=c1, out_channels=64, kernel_size=8, stride=4)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # 2 -> lidar
        self.conv21 = nn.Conv2d(in_channels=c2, out_channels=64, kernel_size=8, stride=4)
        self.conv22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv23 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # 3 -> birdeye
        self.conv31 = nn.Conv2d(in_channels=c3, out_channels=64, kernel_size=8, stride=4)
        self.conv32 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv33 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        out_size = 150528 #TODO: find out

        # Estimation 
        self.dense1 = nn.Linear(out_size, 512)
        self.dense2_2 = nn.Linear(512,256)
        self.dense3_2 = nn.Linear(256,output_dim) # output_dim =1 for

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device('cuda:0')
        self.to(device=self.device)

        # optimal initialization of weights
        self.apply(weights_init_)

    def forward(self, input, action):
        # input_1, _, input_2 = torch.tensor_split(input,(3,3), dim=1)

        # # going through the convolutions
        # input_1 = F.relu(self.conv11(input_1))
        # input_1 = F.relu(self.conv12(input_1))
        # input_1 = F.relu(self.conv13(input_1))
        # input_1 = torch.flatten(input_1,1)

        # input_2 = F.relu(self.conv21(input_2))
        # input_2 = F.relu(self.conv22(input_2))
        # input_2 = F.relu(self.conv23(input_2))
        # input_2 = torch.flatten(input_2,1)


        # # Concatenate conv outputs
        # state_input = torch.cat((input_1, input_2),dim=-1)
        input_1 = input['camera']
        input_2 = input['lidar']
        input_3 = input['birdeye']

        input_1 = F.relu(self.conv11(input_1))
        input_1 = F.relu(self.conv12(input_1))
        input_1 = F.relu(self.conv13(input_1))
        input_1 = torch.flatten(input_1,1)

        input_2 = F.relu(self.conv21(input_2))
        input_2 = F.relu(self.conv22(input_2))
        input_2 = F.relu(self.conv23(input_2))
        input_2 = torch.flatten(input_2,1)

        input_3 = F.relu(self.conv31(input_3))
        input_3 = F.relu(self.conv32(input_3))
        input_3 = F.relu(self.conv33(input_3))
        input_3 = torch.flatten(input_3,1)

        state_input = torch.cat((input_1, input_2, input_3),dim=-1)
        xu = torch.cat([state_input, action], 1)

        # Q1 fully connected forward
        x1 = F.relu(self.dense1_1(xu))
        x1 = F.relu(self.dense2_1(x1))
        x1 = self.dense3_1(x1)
        
        # Q2 fully connected forward
        x2 = F.relu(self.dense1_2(xu))
        x2 = F.relu(self.dense2_2(x2))
        x2 = self.dense3_2(x2)

        return x1, x2
    
    def freeze_paramaters(self):
        """Freeze network parameters"""
        for p in self.parameters():
            p.requires_grad = False
        

    #TODO: check if this still applies to the new Q network
    def save_checkpoint(self, epsilon, num):
        print('... saving checkpoint ...')
        path = self.checkpoint_file / (self.name+'_'+str(num)+'.chkpt')
        torch.save(dict(model=self.state_dict(), epsilon_decay=epsilon), path)
    
    def load_checkpoint(self, checkpoint_file):
        if not checkpoint_file.exists():
            raise ValueError(f"{checkpoint_file} does not exist")
        print('... loading checkpoint ...')
        ckp = torch.load(checkpoint_file)
        exploration_rate = ckp.get('epsilon_decay')
        state_dict = ckp.get('model')
        self.load_state_dict(state_dict)
        return exploration_rate

class PolicyNetwork(nn.Module):
    '''
        Soft Actor Critic Gaussian Policy Network
    '''
    def __init__(self, alpha, input_dim, action_dim, action_space=None, name='PolicyNet', checkpoint='checkpoints/sac'):
        super(PolicyNetwork, self).__init__()
        
        c, h, w = input_dim
        self.name = name
        self.checkpoint_file = checkpoint

        if h != 256:
            raise ValueError(f"Expecting input height: 150, got: {h}")
        if w != 256:
            raise ValueError(f"Expecting input width: 150, got: {w}")
        
        c1 = 3; c2 = 3; c3 = 3# 3 channels lidar, 9 channels rgb camera

        # Feature extraction
        # 1 -> camera
        self.conv11 = nn.Conv2d(in_channels=c1, out_channels=64, kernel_size=8, stride=4)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # 2 -> lidar
        self.conv21 = nn.Conv2d(in_channels=c2, out_channels=64, kernel_size=8, stride=4)
        self.conv22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv23 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # 3 -> birdeye
        self.conv31 = nn.Conv2d(in_channels=c3, out_channels=64, kernel_size=8, stride=4)
        self.conv32 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv33 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        out_size = 150528 #TODO: find out

        # Estimation 
        self.dense1 = nn.Linear(out_size, 512)
        self.dense2 = nn.Linear(512,256)
        self.mean_dense = nn.Linear(256, action_dim)
        self.log_std_dense = nn.Linear(256, action_dim)
        
        # applying initial optimal weights
        self.apply(weights_init_)

        # action rescaling TODO: verify it applies to this case
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)


        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device('cuda:0')
        self.to(device=self.device)

    def forward(self, input: dict):
        '''passing data through policy network'''

        # Extracting them features
        # input_1, _, input_2 = torch.tensor_split(input,(3,3), dim=1)
        input_1 = input['camera']
        input_2 = input['lidar']
        input_3 = input['birdeye']

        input_1 = F.relu(self.conv11(input_1))
        input_1 = F.relu(self.conv12(input_1))
        input_1 = F.relu(self.conv13(input_1))
        input_1 = torch.flatten(input_1,1)

        input_2 = F.relu(self.conv21(input_2))
        input_2 = F.relu(self.conv22(input_2))
        input_2 = F.relu(self.conv23(input_2))
        input_2 = torch.flatten(input_2,1)

        input_3 = F.relu(self.conv31(input_3))
        input_3 = F.relu(self.conv32(input_3))
        input_3 = F.relu(self.conv33(input_3))
        input_3 = torch.flatten(input_3,1)

        state_input = torch.cat((input_1, input_2, input_3),dim=-1)

        x = F.relu(self.dense1(state_input))
        x = F.relu(self.dense2(x))
        mean = self.mean_dense(x)
        log_std = self.log_std_dense(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, input):
        mean, log_std = self.forward(input)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(PolicyNetwork, self).to(device)
    

    def freeze_paramaters(self):
        """Freeze network parameters"""
        for p in self.parameters():
            p.requires_grad = False
        

    # TODO: check if it still applies to this case
    def save_checkpoint(self, epsilon, num):
        print('... saving checkpoint ...')
        path = self.checkpoint_file / (self.name+'_'+str(num)+'.chkpt')
        torch.save(dict(model=self.state_dict(), epsilon_decay=epsilon), path)
    
    def load_checkpoint(self, checkpoint_file):
        if not checkpoint_file.exists():
            raise ValueError(f"{checkpoint_file} does not exist")
        print('... loading checkpoint ...')
        ckp = torch.load(checkpoint_file)
        exploration_rate = ckp.get('epsilon_decay')
        state_dict = ckp.get('model')
        self.load_state_dict(state_dict)
        return exploration_rate


