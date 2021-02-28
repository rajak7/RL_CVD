import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from util import log_normal,sample_gaussian

class encode_structure(nn.Module):
    def __init__(self,net_parameters):
        super(encode_structure,self).__init__()
        self.input_dim = net_parameters['input_dim']
        self.output_dim = net_parameters['output_dim']
        self.linear_1 = nn.Linear(self.input_dim,24)
        self.linear_2 = nn.Linear(24,48)
        self.output = nn.Linear(48,self.output_dim)
    
    def forward(self,x):
        x_size =x.size()
        x = x.reshape(x_size[0],x_size[1],-1)
        h = F.relu(self.linear_1(x))
        h = F.relu(self.linear_2(h))
        x_compressed = F.relu(self.output(h))
        return x_compressed

class decode_structure(nn.Module):
    def __init__(self,net_parameters):
        super(decode_structure,self).__init__()
        self.input_dim = net_parameters['hidden_dim']
        self.output_dim = net_parameters['input_dim']
        self.linear_1 = nn.Linear(self.input_dim,72)
        self.linear_2 = nn.Linear(72,24)
        self.output = nn.Linear(24,3)
        self.sigma = nn.Linear(24,3)

    def forward(self,x):
        h = F.relu(self.linear_1(x))
        h = F.relu(self.linear_2(h))
        out = torch.exp(self.output(h))
        v =  F.softplus(self.sigma(h)) + 1e-8
        return out,v


class model_cvd_series(nn.Module):
    def __init__(self,encoder,decoder,net_parameters,name='default',loss_type='mse',sigma=0.1,sigma_type='const'):
        super(model_cvd_series,self).__init__()
        self.input_dim =  net_parameters['output_dim']
        self.hidden_dim = net_parameters['hidden_dim']
        self.batch_size = net_parameters['batch_size']
        self.encoder = encoder
        self.name = name
        self.lstm = nn.LSTM(input_size=self.input_dim,hidden_size = self.hidden_dim,
                            batch_first=True, bidirectional= False)
        self.hidden = self.init_hidden()
        self.decoder = decoder
        self.loss_type = loss_type
        self.sigma = sigma
        self.sigma_type = sigma_type
        self.nnloss = torch.nn.MSELoss(reduction='none')
    
    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))
    
    def forward(self,x,c):
        x_input = torch.cat((x[:,:-1,:],c[:,:-1,:]),dim=2)
        h_x = self.encoder(x_input)
        h_xt,_ = self.lstm(h_x,self.hidden)
        logits,v = self.decoder(h_xt)
        return logits,v

    def get_hidden(self,x,c):
        x_input = torch.cat((x,c),dim=2)
        #print(x.size(),c.size(),x_input.size())
        h_x = self.encoder(x_input)
        h_xt,_ = self.lstm(h_x,self.hidden)
        return h_xt
    
    def cal_loss(self,logit,v,y):
        target = y[:,1:,:]
        if self.loss_type == 'mse':
            loss = self.nnloss(logit,target)
        else:
            if self.sigma_type == 'const':
                vv = self.sigma * torch.ones(logit.size())
            else:
                vv = v
            loss = -1.0*log_normal(target,logit,vv)  
        loss_tot = loss.sum(-1).sum(-1).mean() 
        if self.loss_type != 'mse' and self.sigma_type != 'const':
            loss_tot = loss_tot + v.sum(-1).sum(-1).mean()    
        return loss_tot
    
    def sample_structure(self,mu,v):
        if self.sigma_type == 'const':
            vv = self.sigma * torch.ones(mu.size())
        else:
            vv = v
        sample = sample_gaussian(mu,vv)
        return sample
    
    def predict_profile(self,conc,y0,return_hx=False):
        Ytemp = torch.zeros((conc.size()[0],conc.size()[1],3))
        Ytemp[:,0,:] = y0
        if conc.size()[1] == 1:
            self.init_hidden()
            h_xt = self.get_hidden(Ytemp,conc)
            logit,v = self.decoder(h_xt)
        else:
            for j in range(conc.size()[1]-1):
                self.init_hidden()
                h_xt = self.get_hidden(Ytemp,conc)
                logit,v = self.decoder(h_xt)
                Ytemp[:,j+1,:] = logit[:,j,:]
        if return_hx == False:
            return logit,v
        else:
            return logit,v,h_xt

    def cal_likelihood(self,x,mu,v):
        if self.sigma_type == 'const':
            vv = self.sigma * torch.ones(mu.size())
        else:
            vv = v
        log_likelihood = log_normal(x,mu,vv)
        return log_likelihood  


def mse_error(y_predicted,y):
    nnloss = torch.nn.MSELoss(reduction='none')
    target = y[:,1:,:]
    loss = nnloss(y_predicted,target)
    return loss

def predict_profile(model,conc,y0,isround=True):
    Ytemp = torch.zeros((conc.size()[0],conc.size()[1],3))
    Ytemp[:,0,:] = y0
    for j in range(conc.size()[1]-1):
        model.init_hidden()
        logit,v=model(Ytemp,conc)
        if isround == True:
            Ytemp[:,j+1,:] = torch.round(logit[:,j,:])
        else:
            Ytemp[:,j+1,:] = logit[:,j,:]
    return logit,v
