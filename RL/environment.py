import torch
import numpy as np 
from numpy import random


#make a class for each state which is [x,c,hx,action=c_next,x_next,h_next,reward]

class state_info():
    def __init__(self,X0,C0,hx0,reward=None,time=None):
        self.X = X0
        self.C = C0
        self.state = hx0
        self.reward = reward
        self.time = time
    
class environment():
    def __init__(self,md_model,reaction_condtion):
        self.md_model = md_model
        self.reaction_condtion = reaction_condtion
        
    def generate_start_condition(self,batch=32):
        return self.reaction_condtion.generate_start_condition(batch)
    
    def generate_next_state(self,c_mean,batch=32,std_temp=250,std_S2=20.0,std_H2=20.0,std_H2S=20.0):
        return self.reaction_condtion.generate_next_state(c_mean,batch,std_temp,std_S2,std_H2,std_H2S)
    
    def generate_profile(self,length=20,batch=32,std_temp=250,std_S2=20.0,std_H2=20.0,std_H2S=20.0):
        return self.reaction_condtion.generate_profile(length,batch,std_temp,std_S2,std_H2,std_H2S)
    
    def generate_range_T_S(self,H2_val,H2S_val,batch=32):
        return self.reaction_condtion.generate_range_T_S(H2_val,H2S_val,batch)
    
    def get_states_representation(self,Cn_,Y0_):
        with torch.no_grad():
            self.md_model.eval()
            logit,v,hx = self.md_model.predict_profile(Cn_,Y0_,return_hx=True)
        return logit,v,hx




