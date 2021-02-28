import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.utils as utils
import numpy as np
import math

class Policy(nn.Module):
    def __init__(self,input_dim,std,dropout=0.5):
        super(Policy, self).__init__()
        self.input_dim = input_dim
        self.action_dim = std.size()[-1]  #torch.tensor(std,dtype=torch.float32).unsqueeze(0)
        self.std = torch.nn.Parameter(std,requires_grad=False)
        self.linear_1 = nn.Linear(self.input_dim,72)
        self.linear_2 = nn.Linear(72,24)
        self.dropout = nn.Dropout(p=dropout)
        self.actor = nn.Linear(24,self.action_dim)
        self.critic = nn.Linear(24,1)
        self.act = F.relu
    
    def forward(self,x):
        inputs = x
        h = self.dropout(self.act(self.linear_1(inputs)))
        s_feature = self.dropout(self.act(self.linear_2(h)))
        mu = self.actor(s_feature)
        v = self.std.expand(x.size()[0], *self.std.shape).reshape(-1, *self.std.shape[1:])
        state_value = self.critic(s_feature) 
        return mu,v,state_value
    
    def set_std(self,std):
        self.std = torch.nn.Parameter(std,requires_grad=False)

class Policy_gradient():
    def __init__(self,input_dim, std,learning_rate=0.0005,gamma=0.99):
        self.model = Policy(input_dim,std)
        self.gamma = gamma
        self.optimizer =optim.Adam(self.model.parameters(), lr=learning_rate)
        self.pi = Variable(torch.FloatTensor([math.pi]))
        self.nnloss = torch.nn.SmoothL1Loss(reduction='none') #torch.nn.MSELoss(reduction='none')
    
    def select_action(self, state):
        mu, v, state_value = self.model(Variable(state))
        action = self.sample_gaussian(mu,v)
        log_prob = self.log_normal(action, mu, v)
        return action, log_prob, mu, state_value
    
    def update_parameters(self, rewards, log_probs,mu_gas,state_value,batch):
        Gi = torch.zeros(batch)
        returns = []
        policy_loss = []
        value_loss = []
        self.optimizer.zero_grad()

        for i in reversed(range(len(rewards))):
            Gi = self.gamma * Gi + rewards[i]     # Monte-Carlo estimate of return 
            returns.insert(0,Gi.unsqueeze(1))

        for n in range(len(rewards)):
            advantage = returns[n] - state_value[n].detach().numpy()
            loss_t = -log_probs[n] * advantage
            value_loss_t = self.nnloss(input=state_value[n],target=returns[n])
            policy_loss.append(loss_t.unsqueeze(1))
            value_loss.append(value_loss_t.unsqueeze(1))
        #print("1",loss_t.size(),value_loss_t.size(),policy_loss[0].size())
        policy_loss = torch.cat(policy_loss,dim=1)   # total reward over an episode
        value_loss = torch.cat(value_loss,dim=1)   # total baseline loss over an episode
        #print("2",policy_loss.size(),value_loss.size())
        policy_loss = policy_loss.sum(-1).sum(-1)
        value_loss = value_loss.sum(-1).sum(-1)
        #print("3",policy_loss.size(),value_loss.size())
        loss = policy_loss.mean()           
        mu_gas = torch.cat(mu_gas)
        loss = loss   + abs(mu_gas).mean()
        v_loss = value_loss.mean()
        tot_loss = loss + v_loss
        tot_loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
        return tot_loss,loss,v_loss
    
    def sample_gaussian(self,mu,v):
        epsilon = torch.normal(torch.zeros(mu.size()),torch.ones(v.size()))
        z = mu + torch.sqrt(v) * epsilon
        return z.data

    def log_normal(self,x,m,v):
        pi_const = 2.0 * np.pi
        log_prob = - ((x-m) ** 2)/(2.0 * v)  - torch.log(torch.sqrt(pi_const * v))
        return log_prob


