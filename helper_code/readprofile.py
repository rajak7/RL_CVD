import sys
import glob
import numpy as np
import os
import subprocess
import shlex, json

def readfile(filename,outfile):
    Natoms = 0
    data = []
    with open(filename,'r') as in_file:
        for val in in_file:
            Natoms += 1
            data.append(val.strip())
    with open(outfile,'w') as out_file:
        out_file.write(str(Natoms) + '\n')
        out_file.write('50.990  49.000  94.000'+'\n')
        for val in data:
            temp_data = val.split()
            out_file.write(temp_data[2]+' '+temp_data[4]+' '+temp_data[5]+' '+temp_data[6]+'\n')
        out_file.close()


def findfile_list(dir_val):
     time_range = np.arange(1,21)
     sequence_number = dir_val.split('_')[-1]

     #print(dir_val, sequence_number)

     os.system('rm -rf %s' %('/home/rcf-40/ankitmis/staging/ML_CVD/' + sequence_number))
     os.system('mkdir %s' %('/home/rcf-40/ankitmis/staging/ML_CVD/' + sequence_number))

     T_data = '/home/rcf-40/ankitmis/staging/ML_CVD/' + sequence_number +'/Training_Data'
     T_data_json = '/home/rcf-40/ankitmis/staging/ML_CVD/' + sequence_number +'/Training_Data/JSON'

     os.system('rm -rf %s' %T_data)
     os.system('mkdir %s' %T_data)
     
     os.system('rm -rf %s' %T_data_json)
     os.system('mkdir %s' %T_data_json)


     profile = open(dir_val + '/profile', 'r')
     profiles = []
     for line in profile.readlines():
         line = line.strip().split()
         profiles.append([ float(line[0]), float(line[1]), float(line[2]), float(line[3]) ])

     for time in time_range:
         subdir_list = glob.glob(dir_val + '/ARCHIVE/temp-'+str(time)+'-*/')
         for subdir in subdir_list:
             input_file = subdir+'000000000.pdb'
             out_file =T_data+'/'+str(time)+'.xyz'
             out_tensor = T_data+'/'+str(time)+'.txt'
             print(input_file,out_file)
             readfile(input_file,out_file)
             input="./2d_phase"+" "+ out_file+" "+out_tensor
             args=shlex.split(input)
             child_process=subprocess.Popen(args)

             """
             if time == 20:
                 input_file = subdir+'000900000.pdb'
                 out_file =T_data+'/'+str(21)+'.xyz'
                 out_tensor = T_data+'/'+str(21)+'.txt'
                 print(input_file,out_file)
                 readfile(input_file,out_file)
                 input="./2d_phase"+" "+ out_file+" "+out_tensor
                 args=shlex.split(input)
                 child_process=subprocess.Popen(args)
             """
             child_process.wait()
             if child_process.returncode > 0 :
                 print("-----error-------")
                 sys.exit()
             else:
                 print("----Successful-----")
     
     for time in time_range:
         tensor_file = T_data+'/'+str(time)+'.txt'
         lines = open(tensor_file, 'r').readlines()
         lines = lines[1:]
         a1 = dict()
         b1 = dict()
         l = []
         for line in lines:
             line = line.strip().split()
             l1 = []
             for a in line:
                 l1.append(float(a))
             l.append(l1)

         k = str( sequence_number + " " + str(time))
         b1[k ] = profiles[time-1]
         a1[k ] = l
         json_file1 = T_data_json+'/'+str(time)+'_tensor.json'
         json_file2 = T_data_json+'/'+str(time)+'_profile.json'


         with open(json_file1, 'w') as f:
             json.dump(a1, f)

         with open(json_file2, 'w') as f:
             json.dump(b1,f)

         """
         a1 = dict()
         b1 = dict()
         tensor_file = T_data+'/'+str(21)+'.txt'
         lines = open(tensor_file,'r').readlines()
         lines = lines[1:]
         l = []
         json_file1 = T_data_json+'/'+str(21)+'_tensor.json'
         json_file2 = T_data_json+'/'+str(21)+'_profile.json'
         for line in lines:
             line = line.strip().split()
             l1 = []
             for a in line:
                 l1.append(float(a))
             l.append(l1)

         k = str( sequence_number + " " + str(21))
         a1[k ] = l
         b1[k] = profiles[19]
 
         with open(json_file1, 'w') as f:
             json.dump(a1, f)

         with open(json_file2, 'w') as f:
             json.dump(b1,f)
         """
dirlist = glob.glob('/staging/pv/kris658/MLCVD/DATA/MLCVD_*')
dirlist = np.array(dirlist)
#print(dirlist)
counter = 1
for val in dirlist:
    #print(counter, 'reading directory: ',val)

    #if val == '/staging/pv/kris658/MLCVD/DATA/MLCVD_01474':
    #    continue
    #if val == '/staging/pv/kris658/MLCVD/DATA/MLCVD_05410':
    #    continue

    findfile_list(val)


