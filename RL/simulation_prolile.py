import numpy as np
from numpy import random

class reaction_condition():
    def __init__(self,T_range=[750.0,3750.0],S2=[20.0,250.0],H2=[20.0,250.0],H2S=[20.0,200.0]):
        self.T_range = T_range
        self.S2 = S2
        self.H2 = H2
        self.H2S = H2S
    
    def generate_profile(self,length=20,batch=32,std_temp=250,std_S2=20.0,std_H2=20.0,std_H2S=20.0):
        conditon = np.zeros((batch,length,4),dtype=float)
        conditon[:,0,:] = self.generate_start_condition(batch=batch)
        for n in range(1,length):
            conditon[:,n,:] = self.generate_next_state(conditon[:,n-1,:],batch,std_temp,std_S2,std_H2,std_H2S)
        return conditon
    
    def generate_start_condition(self,batch=32):
        conditon = np.zeros((batch,4),dtype=float)
        conditon[:,0] = self.T_range[0] + (self.T_range[1]- self.T_range[0]) * np.random.random(batch)
        conditon[:,1] = self.S2[0] + (self.S2[1]- self.S2[0]) * np.random.random(batch)
        conditon[:,2] = self.H2[0] + (self.H2[1]- self.H2[0]) * np.random.random(batch)
        conditon[:,3] = self.H2S[0] + (self.H2S[1]- self.H2S[0]) * np.random.random(batch)
        #conditon = np.expand_dims(conditon, axis=1)
        return conditon
    
    def generate_range_T_S(self,H2_val,H2S_val,batch=32):
        conditon = np.zeros((batch,4),dtype=float)
        del_T = (self.T_range[1]- self.T_range[0]) / 4.0
        del_S = (self.S2[1]- self.S2[0]) / 10.0
        #print("delT and delS",del_T,del_S)
        for t in range(4):
            T_val = self.T_range[0] + (t+1) *del_T
            for s in range(8):
                S_val = self.S2[0] + s * del_S
                conditon[s+8*t,0] = T_val
                conditon[s+8*t,1] = S_val
                conditon[s+8*t,2] = H2_val
                conditon[s+8*t,3] = H2S_val
        return conditon
    
    def generate_next_state(self,c_mean,batch=32,std_temp=250,std_S2=20.0,std_H2=20.0,std_H2S=20.0):
        conditon = np.zeros((batch,4),dtype=float)
        #temperature
        t1 = np.random.normal(c_mean[:,0],std_temp,size=batch)
        t1[t1 < self.T_range[0]] = self.T_range[0]
        t1[t1 > self.T_range[1]] = self.T_range[1]
        conditon[:,0] = np.copy(t1)
        #S1
        s1 = np.random.normal(c_mean[:,1],std_S2,size=batch)
        s1[s1 < self.S2[0]] = self.S2[0]
        s1[s1 > self.S2[1]] = self.S2[1]
        conditon[:,1] = np.copy(s1)
        #H2
        h1 = np.random.normal(c_mean[:,2],std_H2,size=batch)
        h1[h1 < self.H2[0]] = self.H2[0]
        h1[h1 > self.H2[1]] = self.H2[1]
        conditon[:,2] = np.copy(h1)
        #H2S
        hs1 = np.random.normal(c_mean[:,3],std_H2S,size=batch)
        hs1[hs1 < self.H2S[0]] = self.H2S[0]
        hs1[hs1 > self.H2S[1]] = self.H2S[1]
        conditon[:,3] = np.copy(hs1)
        hs = hs1
        return conditon

        








