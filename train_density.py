import numpy as np 
import torch
from torch.utils import data
from dataloader import CVD_Dataset
from desity_estimator import encode_structure, model_cvd_series, decode_structure
from util import save_model
import sys
import os

model_name = sys.argv[1]
num_epochs = 1000
sigma_type = 'const'
sigma = 0.01
loss_type='mle'
#model parameters 
net_parameters = {'input_dim':7,'output_dim':72,'hidden_dim':128,'batch_size':32}

print("model_name: ",model_name)

path = os.path.join('checkpoints',model_name)
if not os.path.exists(path):
    os.makedirs(path)
log_file = os.path.join(path,'log.txt')
log = open(log_file,'w')

#load training data
Structure_train_np = np.load('Data/Train/CVD_Structure.npy')
Condition_train_np = np.load('Data/Train/CVD_Condition.npy')
Structure_train = torch.tensor(Structure_train_np,dtype=torch.float32)
Condition_train = torch.tensor(Condition_train_np,dtype=torch.float32)
N_train = Structure_train_np.shape[0]
print("Number of training examples: ",N_train)
#data loader for training data
train_data = CVD_Dataset(structure=Structure_train,condtion=Condition_train)
train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)

#model traininig parameters 
encoder=encode_structure(net_parameters)
decoder = decode_structure(net_parameters)
model = model_cvd_series(encoder,decoder,net_parameters,loss_type=loss_type,sigma=sigma,sigma_type=sigma_type)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#train the model
print("=====Training Starts======")
tot_loss = []
for n in range(num_epochs):
    model.train()
    for i,batch in  enumerate(train_loader):
        S_ = batch['structure']
        C_ = batch['condtion']
        Y_ = batch['P_class']
        optimizer.zero_grad()
        model.zero_grad()
        model.init_hidden()
        logit,v=model(Y_,C_)
        loss = model.cal_loss(logit,v,Y_)
        loss.backward()
        optimizer.step()
        tot_loss.append(loss.item())
        if i % 50 == 0:
            print("epoch: {0:12d}  step: {1:12d}  loss = {2:12.6f}".format(n,i,loss.item()))
            log.write("epoch: {0:12d}  step: {1:12d}  loss = {2:12.6f} \n".format(n,i,loss.item()))
#save final model
save_model(model,model_name)
#save training loss 
np.save(path+'/'+'tot_loss',tot_loss)
print("=====Training Ends======")

