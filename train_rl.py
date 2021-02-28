import torch
import numpy as np
from RL.environment import environment,state_info
from RL.rl_net import Reinforce
from desity_estimator import encode_structure,model_cvd_series,decode_structure
from util import load_model

#load MD surrogate model
model_name = 'MD_MoS2'
net_parameters = {'input_dim':7,'output_dim':72,'hidden_dim':128,'batch_size':32}
encoder=encode_structure(net_parameters)
decoder = decode_structure(net_parameters)
model = model_cvd_series(encoder,decoder,net_parameters,loss_type='mle',sigma=0.01)
#load moldel parameters
load_model(model,model_name)
print("sigma_type:",model.sigma_type,"loss_type:",model.loss_type,"sigma:",model.sigma)

#create RL agrent 
sim_param_std = torch.tensor([250.0,20.0,20.0,20.0],dtype=torch.float32).unsqueeze(0)
rl_agent = Reinforce(net_parameters['hidden_dim'],sim_param_std,learning_rate=0.001,gamma=0.99)