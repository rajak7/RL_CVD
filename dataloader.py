
import numpy as np
import os
import json
import torch 
from torch.utils import data


#read data and create a bigger tensor
def read_cvd_numpy_data(i_start,i_end,per_batch_file,file_dir='Data'):
    #X=np.asarray(X).reshape(-1,X_1.shape[1],X_1.shape[2],X_1.shape[3])
    #Y=np.asarray(Y).reshape(-1,Y_1.shape[1],Y_1.shape[2])
    ngroup = int((i_end-i_start)/per_batch_file)
    print("total group of files to be read: ",ngroup)
    cur_start = i_start
    cur_end = cur_start + per_batch_file
    for n in range(ngroup):
        fname = str(cur_start)+'_'+str(cur_end)
        print("reading dir: ",fname)
        x_name = os.path.join(file_dir,"Structure_"+fname+'.npy')
        y_name = os.path.join(file_dir,"Condition_"+fname+'.npy')
        x_i = np.load(x_name)
        y_i = np.load(y_name)
        if n == 0:
            X_all = x_i
            Y_all = y_i
        else:
            X_all = np.concatenate((X_all,x_i),axis=0)
            Y_all = np.concatenate((Y_all,y_i),axis=0)
    return X_all,Y_all

#create a batch profile of data
def create_data_batch(filedir,i_start,i_end,per_batch_file,savedir='Data'):
    ngroup = int((i_end-i_start)/per_batch_file)
    cur_start = i_start
    cur_end = cur_start + per_batch_file
    for n in range(ngroup):
        fname = str(cur_start)+'_'+str(cur_end)
        print("reading dir: ",fname)
        x, y = data_loader(filedir,cur_start,cur_end)
        x_name = os.path.join(savedir,"Structure_"+fname)
        y_name = os.path.join(savedir,"Condition_"+fname)
        np.save(x_name,x)
        np.save(y_name,y)
        cur_start=cur_end
        cur_end = cur_start+per_batch_file
    return

#read JSON file of simulation condtion and simulation profile
def data_loader(filedir,i_start,i_end,trj_len=20):
    x = [] # concentation
    y = [] # simulation condition
    for i in range(i_start,i_end):
        filename = os.path.join(filedir,str(i).zfill(5),'Training_Data/JSON')
        profile = []
        for j in range(1,21):
            x_name = os.path.join(filename,str(j)+'_tensor.json')
            y_name = os.path.join(filename,str(j)+'_profile.json')
            # read the simulation conditions first
            with open (y_name,'r') as in_file:
                d = json.load(in_file)
                for val in d.values():
                    y.append(val)
            # read the simulation output
            with open (x_name,'r') as in_file:
                d_x = json.load(in_file)
                temp_x = []
                for val_x in d_x.values():
                    temp_x.append(val_x)
                profile.append(temp_x)
        pp = np.asarray(profile).squeeze(1)
        #print(i,'has unique values: ',np.unique(pp))
        x.append(pp)            
    n_files = i_end-i_start
    x = np.asarray(x)
    y=np.asarray(y).reshape(-1,trj_len,len(val))
    if n_files != y.shape[0]  or n_files != x.shape[0]:
        print("error in ",x.shape,y.shape,n_files)
        raise exception ("x.shape or y.shape does not match")
    return x, y


#create queantity of each pahse in the structure 
class CVD_Dataset(data.Dataset):
    def __init__(self,structure,condtion,natoms=288.0):
        n_data = structure.size(0)   #simulation number
        t_data = structure.size(1)   #trajectory number
        print("total simulations and trajectory: ",n_data,t_data)
        self.structure = torch.zeros((n_data,t_data,3),dtype=torch.float32)    # no. of 2H,1T, defect atoms
        self.p_structure = torch.zeros((n_data,t_data,3),dtype=torch.float32)  # percentage of 2H,1T and defect atoms
        self.atoms_profile = structure    #atoms location
        self.condtion = condtion          #reaction condition
        for i in range(n_data):
            for j in range(t_data):
                _2H    = torch.sum(structure[i,j] == 1)
                _1T    = torch.sum(structure[i][j] == 2)
                defect = torch.sum(structure[i][j] == 3)
                self.structure[i,j,0] =  _2H
                self.structure[i,j,1] =  _1T
                self.structure[i,j,2] =  defect
                self.p_structure[i,j,0] =  _2H / natoms
                self.p_structure[i,j,1] =  _1T / natoms
                self.p_structure[i,j,2] =  defect / natoms
    
    def __len__(self):
        return self.structure.size(0)

    def __getitem__(self, index):
        S_ = self.structure[index]     #no. of atoms of each type
        C_ = self.condtion[index]      #simulation condition
        P_ = self.p_structure[index]   #density of atoms of each type
        Atoms = self.atoms_profile[index]  # atoms coordinates in the simulation
        return {'structure':S_,'condtion':C_,'P_class':P_,'Atoms':Atoms}
