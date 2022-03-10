# 12 channel : 3 lidar + 9 rgb


import torch
from torch import nn
from torch._C import device 
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, alpha, input_dim, output_dim, name='online', checkpoint='checkpoints/dqn'):
        super(Network, self).__init__()
        
        c, h, w = input_dim
        self.name = name
        self.checkpoint_file = checkpoint

        if h != 150:
            raise ValueError(f"Expecting input height: 150, got: {h}")
        if w != 150:
            raise ValueError(f"Expecting input width: 150, got: {w}")
        
        c1 = 3; c2 = 9

        
        self.conv11 = nn.Conv2d(in_channels=c1, out_channels=64, kernel_size=8, stride=4)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.conv21 = nn.Conv2d(in_channels=c2, out_channels=64, kernel_size=8, stride=4)
        self.conv22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv23 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.dense1 = nn.Linear(28800, 512)
        self.dense2 = nn.Linear(512,256)
        self.q_values = nn.Linear(256,output_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device('cuda:0')
        self.to(device=self.device)
    
    def freeze_paramaters(self):
        """Freeze network parameters"""
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, input):
        input_1, _, input_2 = torch.tensor_split(input,(3,3), dim=1)

        input_1 = F.relu(self.conv11(input_1))
        input_1 = F.relu(self.conv12(input_1))
        input_1 = F.relu(self.conv13(input_1))
        input_1 = torch.flatten(input_1,1)

        input_2 = F.relu(self.conv21(input_2))
        input_2 = F.relu(self.conv22(input_2))
        input_2 = F.relu(self.conv23(input_2))
        input_2 = torch.flatten(input_2,1)

        input = torch.cat((input_1, input_2),dim=-1)
        input = F.relu(self.dense1(input))
        input = F.relu(self.dense2(input))
        input = self.q_values(input)
        # input = torch.tanh(self.q_values(input))
        return input
        
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