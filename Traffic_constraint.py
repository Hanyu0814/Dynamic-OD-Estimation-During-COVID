#!/usr/bin/env python
# coding: utf-8

# ## Read data

# In[1]:


import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import pickle
from collections import OrderedDict
import copy
from scipy.sparse import csr_matrix
from scipy import io
import seaborn as sns
import joblib
from base import *
from joblib import Parallel, delayed
import random
import scipy as sc
import sys


# In[2]:


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# ### Read raw speed and count data

# In[3]:

start_month=int(sys.argv[1])
start_day=int(sys.argv[2])
end_month=int(sys.argv[3])
end_day=int(sys.argv[4])
year =int(sys.argv[5])
flag_weekday = int(sys.argv[6])


read_in_folder = '../../../Raw_data_hpc/Processed_data_networkL/'
read_in_pkl = '../../../Raw_data_hpc/link_dict/'
write_folder = './'


with open(read_in_pkl+'volume_dict_link_'+str(year)+"_"+str(start_month)+"_"+str(start_day)+'.pkl', 'rb') as handle:
    count_data = pickle.load(handle)
with open(read_in_pkl+'speed_dict_link_'+str(year)+"_"+str(month)+"_"+str(day)+'.pkl', 'rb') as handle:
    spd_data = pickle.load(handle)
    
print(spd_data)


# ### Read graph data

# In[5]:


with open(read_in_folder+'od_list.pickle', 'rb') as handle:
    (O_list, D_list) = pickle.load(handle)
# with open('graph.pickle', 'rb') as handle:
#     G = pickle.load(handle)
G = nx.read_gpickle(read_in_folder+'graph_4.pickle')

# with open(read_in_folder+'od_list_4.pickle', 'rb') as handle:
#     pickle.dump((O_list, D_list),handle, protocol=4)


# In[6]:


with open(read_in_folder+'od_list.pickle', 'rb') as handle:
    (O_list, D_list) = pickle.load(handle)
# with open('graph.pickle', 'rb') as handle:
#     G = pickle.load(handle)
G = nx.read_gpickle(read_in_folder+'graph_4.pickle')
G = nx.freeze(G)
# print(O_list)
pos=nx.get_node_attributes(G,'pos')
plt.rcParams['figure.figsize'] = [8, 6]
nx.draw_networkx(G,pos,node_size=10,node_color='blue',with_labels=False)
# print(G.number_of_edges())
# print(G.number_of_nodes())
Node = list(G.nodes)
Node.sort()
# print("Nodes: ", Node)

print(O_list)

# In[7]:


#no > flag_weekday1 & no < flag_weekday2:
if flag_weekday == 1:
    flag_weekday1 = -1
    flag_weekday2 = 7
else:
    flag_weekday1 = -1
    flag_weekday2 = 7

start_date = datetime.date(year, start_month, start_day)
end_date = datetime.date(year, end_month, end_day)
delta_date = datetime.timedelta(days = 1)
time_basis = datetime.time(0,0,0)
N_path = 1
time_interval_min = 5


# In[8]:


from itertools import islice
def k_shortest_paths(G, source, target, k, weight=None):
    def path_cost(G, path):
        return sum([G[path[i]][path[i+1]]['length'] for i in range(len(path)-1)])
    p = list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
    return p


# In[9]:


OD_paths = OrderedDict()
link_dict = OrderedDict()
path_list = list()
shortest_path_list = list()
count = 0;
k=1
N_path = 1
# while len(shortest_path_list) < N_path:
OD_list_output = []
print(O_list)
for O in O_list:
    for D in O_list:
        print(O, D);
        if O!=D:
            try:
                paths = k_shortest_paths(G,O, D,N_path,weight='length') #My code
                if len(paths) != 0:
                    tmp_path_list = list()
                    for path in paths:
                        path_o = Path()
                        path_o.node_list = path
                        path_o.node_to_list(G, link_dict)
                        tmp_path_list.append(path_o)
                        path_list.append(path_o)
                        
                    OD_paths[(O, D)] = tmp_path_list
                    shortest_path_list.append(paths)
                    OD_list_output.append([O,D])
                    print("From ", O, " To ", D, "there are ", k, "paths")
                else:
                    pass
            except:
#                     print("From ", O, " To ", D, "there are no paths")
                pass
        else:
#                 print("From ", O, " To ", D, "same")
            pass
np.savetxt('path_'+start_date.strftime("%m_%d_%Y")+'_'+           str(time_interval_min)+'.txt', OD_list_output, delimiter = ',',fmt='%s')


# Find k shortest path:
# 1. for each pair of OD find shortest path
# 2. order all shortest paths
# 3. find the top k shortest path among all

# ## Generate Delta and Set Parameters


analysis_start_time = datetime.time(0, 0, 0)
time_interval = datetime.timedelta(minutes=time_interval_min)
num_OD = len(OD_paths)
link_list = list(link_dict.values())
num_link = len(link_list)
num_path_v = [len(x) for x in OD_paths.values()]
num_path = np.sum(num_path_v)
N = int(60 / time_interval_min * 24)-1

assert(len(path_list) == num_path)


# In[11]:

delta = np.zeros((num_link, num_path))
for i, link in enumerate(link_list):
    for j, path in enumerate(path_list):
        if link in path.link_list:
            delta[i,j] = 1.0


link_loc = dict()
for idx, link in enumerate(link_list):
    link_loc[link] = idx


# ## Build assignment matrix

# In[13]:


cur_date_time = datetime.datetime.combine(start_date, time_basis)
end_date_time = datetime.datetime.combine(end_date, time_basis)

date_need_to_finish = list()

while(cur_date_time <= end_date_time):
    no = cur_date_time.weekday()

    if (no > flag_weekday1) & (no < flag_weekday2):
        single_date = cur_date_time.date()
        date_need_to_finish.append(single_date)
    cur_date_time = cur_date_time + delta_date
print(date_need_to_finish)


# A parallel computing framework is used to compute the R matrix as well as P matrix. Since we have a 8 core CPU, so we use 7 process to run the program, leaving one core to ensure the desktop does not get stuck.

# In[14]:


import importlib
importlib.reload(sys.modules['base'])
Parallel(n_jobs=3, temp_folder = 'temp', max_nbytes = '10M')(delayed(save_r)(N, spd_data, analysis_start_time, time_interval, 
                        tmp_date, link_dict, link_list, link_loc, path_list) for tmp_date in date_need_to_finish)


# ## Construct P matrix

# In[15]:


cur_date_time = datetime.datetime.combine(start_date, time_basis)
end_date_time = datetime.datetime.combine(end_date, time_basis)

date_need_to_finish = list()

while(cur_date_time <= end_date_time):
# #     date_need_to_finish.append(cur_date_time)
    no = cur_date_time.weekday()
    if (no > flag_weekday1) & (no < flag_weekday2):
        single_date = cur_date_time.date()
        date_need_to_finish.append(single_date)
    cur_date_time = cur_date_time + delta_date


# #### parallel computing

# In[16]:


start_time = datetime.datetime.now()
Parallel(n_jobs=3)(delayed(save_p)(N, spd_data, analysis_start_time, time_interval, 
                                   tmp_date, path_list, OD_paths) for tmp_date in date_need_to_finish)
t_ = (datetime.datetime.now() - start_time)
print("solve time:", t_)


# ## Construct link flow vector

# In[17]:


o_link_list = list(filter(lambda x: x.ID in count_data.keys(), link_list))


# In[18]:


def nearest(tmp_date, items, pivot):
    return min(items, key=lambda x: abs(datetime.datetime.combine(tmp_date,x) - datetime.datetime.combine(tmp_date,pivot)))

def get_x_o(N, o_link_list, tmp_date, analysis_start_time, time_interval, count_data):
#     tmp_date = start_date
    num_o_link = len(o_link_list)
#     print("num: ", num_o_link)
    x = np.zeros(num_o_link * N)
#     x = np.zeros(N)
    #print("x: ", x)
    for h in range(N):
#         tmp_date = datetime.datetime.combine(tmp_date, analysis_start_time)
        start_time = (datetime.datetime.combine(tmp_date, analysis_start_time) + h * time_interval).time()
        end_time = (datetime.datetime.combine(tmp_date, analysis_start_time) + (h+1) * time_interval).time()
        for a, link in enumerate(o_link_list):
            possible_time = np.array(count_data[link.ID].columns)
            print('ID', link.ID);
            if count_data[link.ID].shape[0] == 0:
                if h != N-1:
                    x[h * num_o_link + a] = 0
                else:
                    x[h * num_o_link + a] = 0
            else:
                df = count_data[link.ID].loc[tmp_date,:]
                if h != N-1:
                    x[h * num_o_link + a] = df.loc[(df.index.values >= start_time) & 
                                                (df.index.values <= end_time)].sum()
                else:
                    x[h * num_o_link + a] = df.loc[(df.index.values >= start_time)].sum()
#         print('x, ', x)
    date_str = tmp_date.strftime("%Y-%m-%d")
    np.save(os.path.join('X_vector', date_str), x)
    return x


# In[19]:


cur_date_time = datetime.datetime.combine(start_date, time_basis)
end_date_time = datetime.datetime.combine(end_date, time_basis)

while(cur_date_time <= end_date_time):
    no = cur_date_time.weekday()
    if (no > flag_weekday1) & (no < flag_weekday2):
        single_date = cur_date_time.date()
        date_need_to_finish.append(single_date)
        x = get_x_o(N, o_link_list, single_date, analysis_start_time, time_interval, count_data)
        date_str = cur_date_time.strftime("%Y-%m-%d")

    cur_date_time = cur_date_time + delta_date


# ## Construct sysmmetric constraints

# In[20]:


# Decide the OD region.
node_df = pd.read_csv(read_in_folder + 'node_list.csv')
regions = np.linspace(1, 26, num=26)
# print(node_df)

node_df[node_df['node_id'] == 1]['Region']


# In[21]:


regions = np.linspace(1, 26, num=26)
num_regions = len(regions)

# for tmp_date in date_need_to_finish: 
row_list = []
col_list = []
data_list =[]

# num_origin = len(origin_list)
y_loc = 0
print(num_regions * num_regions/2)
Alr_sym = np.zeros((int(num_regions * num_regions), num_OD * N))

for h in range(N):
    rs = 0
    k = 0
    a = -1
    
    for (O,D), paths in OD_paths.items():
#         print('O, ', O,' D, ', D)
        x_loc = h * num_OD + rs # OD list
        rs+=1
        O_index = node_df[node_df['node_id'] == O].Region.iloc[0]
#         print('O_index, ',O_index)

        D_index = node_df[node_df['node_id'] == D].Region.iloc[0]
#         print('D_index, ',D_index)
        if (O_index == D_index):
            continue
        elif (O_index < D_index):
            y_loc = (O_index-1)*num_regions+D_index
            row_list.append(y_loc)
            col_list.append(x_loc)
            Alr_sym[y_loc, x_loc] = 1
            data_list.append(1)
        elif (O_index > D_index):
            y_loc = (D_index-1)*num_regions+O_index
            row_list.append(y_loc)
            col_list.append(x_loc)
            Alr_sym[y_loc, x_loc] = -1
            data_list.append(-1)


print('Alr, ', Alr_sym, 'Alr_tmp nonzero elements, ', np.count_nonzero(Alr_sym))
print('Alr shape, ', Alr_sym.shape)
b_sym = np.zeros(num_regions * num_regions)

with open('Constraints/Alr_sym.npy', 'wb') as f:
    np.save(f, Alr_sym)
with open('Constraints/b_sym.npy', 'wb') as f:
    np.save(f, b_sym)


# ## Add arterial constraint

# In[22]:

read_folder = '../Raw_data/Processed_data/arterial_2020/vol_all.pkl'
with open(read_folder, 'rb') as handle:
    vol_arterial = pickle.load(handle);

regions = np.linspace(1, 26, num=26)
num_regions = len(regions)

# for tmp_date in date_need_to_finish: 
row_list = []
col_list = []
data_list = []

# num_origin = len(origin_list)
y_loc = 0
print(len(Node))
Alr_arterial = np.zeros((len(Node), num_OD * N))
b_arterial = np.zeros((len(Node),1))
for h in range(N):
    rs = 0
    k = 0
    a = -1
    for (O,D), paths in OD_paths.items():
        x_loc = h * num_OD + rs # OD list
        rs+=1

        if (O in vol_arterial.keys()):
#             print('O, ',O)
            row_list.append(O)
            col_list.append(x_loc)
            Alr_arterial[O, x_loc] = 1
            data_list.append(1)
            b_arterial[O] = vol_arterial[O]
            
        if (D in vol_arterial.keys()):
#             print('D, ',D)
            row_list.append(D)
            col_list.append(x_loc)
            Alr_arterial[D, x_loc] = 1
            data_list.append(1)
            b_arterial[D] = vol_arterial[D]
        
Alr_arterial = Alr_arterial[~np.all(Alr_arterial == 0, axis=1)]
b_arterial = b_arterial[~np.all(b_arterial == 0, axis=1)]

Alr_arterial_I = -np.identity(len(Alr_arterial))
Alr_arterial =np.concatenate((Alr_arterial, Alr_arterial_I), axis=1)

with open('Constraints/Alr_arterial.npy', 'wb') as f:
    np.save(f, Alr_arterial)
with open('Constraints/b_arterial.npy', 'wb') as f:
    np.save(f, b_arterial)


# ## Create the observed delta (time dependent)

# In[23]:


observe_index = np.fromiter(map(lambda x: x in o_link_list, link_list),dtype=np.int)
observe_index_N = np.tile(observe_index, (N,))
np.save(os.path.join("observe_index_N"), observe_index_N)


# In[24]:


print(sum(observe_index_N == 1))
print(len(observe_index_N))


# ## Load data to conduct DODE

# from pfe import nnls

# In[25]:

date_str = start_date.strftime("%Y-%m-%d")


# In[26]:


cur_date_time = datetime.datetime.combine(start_date, time_basis)
end_date_time = datetime.datetime.combine(end_date, time_basis)

date_need_to_finish = list()

while(cur_date_time <= end_date_time):
    no = cur_date_time.weekday()

    if (no > flag_weekday1) & (no < flag_weekday2):
        single_date = cur_date_time.date()
        date_need_to_finish.append(single_date)
    cur_date_time = cur_date_time + delta_date
print(date_need_to_finish)


# In[27]:


def split_select(A, mask, N):
    A_list = np.array_split(A, N)
    print(len(A_list))
    mask_list = np.array_split(mask, N)
    tmp_list = []
    for i in range(len(A_list)):
        tmp_A = A_list[i]
        print("tmp_A shape, ", tmp_A.shape)
        tmp_mask = mask_list[i]
        print("tmp_mask, ", tmp_mask.shape)
        A_mask = tmp_A[tmp_mask == 1, :]
        print("shape A_mask, ", A_mask.shape)
        tmp_list.append(A_mask)
#         break
    print("concatenate")
    A = np.concatenate(tmp_list, axis = 0)
    print("shape A, ", A.shape)
    return A

# ## Initial q vector

# In[28]:


np.random.seed(1)
regions = np.linspace(1, 26, num=26)
# num_regions = len(regions)
M = len(OD_paths)
# for tmp_date in date_need_to_finish: 
row_list = []
col_list = []
data_list =[]

# num_origin = len(origin_list)
y_loc = 0
q_sym = np.zeros((M, N))
dict_q = {}

for i,((O,D), paths) in enumerate(OD_paths.items()):
    if (O,D) in dict_q.keys():
        q_sym[i,:] = dict_q[(O,D)];
    else:
        tmp = np.random.rand(1,N);
        dict_q[(O,D)] = tmp;
        dict_q[(D,O)] = tmp;
        q_sym[i,:] = tmp;
        
q_sym = q_sym.T.reshape(-1)
print(q_sym.shape)


# ## NNLS model

# In[37]:


import importlib
from pfe_sym import nnls
importlib.reload(sys.modules['pfe_sym'])
import math

def solve_nnls(date, sum_val = 0, penalty_beta = 0, penalty_gamma = 0):
    cur_date_time = datetime.datetime.combine(start_date, time_basis)

    all_r = []
    all_P = []
    all_x_o = np.array([])
    all_alr = []
    all_A = None

    q_dict = {}
    q_dict_2d = {}

    N = int(60 / time_interval_min * 24)
    M = len(OD_paths)

    i = 0
    observe_index_N = np.load("observe_index_N.npy")

#     while(cur_date_time <= end_date_time):
#         print(cur_date_time)
    no = cur_date_time.weekday()
    if (no > flag_weekday1) & (no < flag_weekday2):
        single_date = cur_date_time.date()
        date_str = single_date.strftime("%Y-%m-%d")

        x_o = np.load(os.path.join('X_vector', date_str + ".npy"))
#         print('X_vector', date_str," ",x)

        print('x: ',len(x_o))
        r = joblib.load(os.path.join("R_matrix", date_str + ".pickle")).tocsr()
#         r = joblib.load(os.path.join("R_matrix", date_str + ".pickle")).toarray()
        print('r: ', r.shape)
        print('r type: ', type(r))

        P = joblib.load(os.path.join("P_matrix", date_str+".pickle")).tocsr()
#         P = joblib.load(os.path.join("P_matrix", date_str+".pickle")).toarray()
        print('P: ',P.shape)

        RP = np.array(r.dot(P).toarray())[observe_index_N == 1,:]
#         RP = (r.dot(P)).toarray()
#         print('RP: ', RP.shape)
#         print('RP: ', type(RP))
#         print('observe_index_N: ', observe_index_N.shape)

        ###############################################
        ################Read in file###################
        ###############################################
        Alr_sym = np.load(os.path.join('Constraints', "Alr_sym.npy"))
        Alr_arterial = np.load(os.path.join('Constraints', "Alr_arterial.npy"))
        b_sym = np.load(os.path.join('Constraints', "b_sym.npy"))
        b_arterial = np.load(os.path.join('Constraints', "b_arterial.npy"))
        ###############################################
        ###############################################

        ###############################################
        ################Constraints####################
        ###############################################
        np.random.seed(2)

        x2,y2 = Alr_arterial.shape
        A = np.concatenate((RP, Alr_sym*penalty_beta), axis=0)
        x1,y1 = A.shape
        A = np.concatenate((A, np.zeros((x1,y2-y1))), axis=1)
        A = np.concatenate((A, Alr_arterial), axis=0)

        b = np.concatenate((x_o, b_sym), axis=0)
        b = np.concatenate((b, b_arterial.reshape(-1)), axis = 0)
        b_max = np.max(b)
        q_init = np.random.rand(A.shape[1])
#         q_init = np.concatenate((q_sym, np.random.rand(y2-y1)), axis=0)
        print('q shape, ', q_init.shape)
#         b = b/b_max
        ###############################################
        ###############################################

        ###############################################
        #########Error upper bound constraint##########
        ###############################################
        A_ones = np.ones((1,RP.shape[1]))        
        A_errorUb = np.concatenate((A_ones, np.zeros((1,y2-y1))), axis=1)
        A = np.concatenate((A, -A_errorUb*penalty_gamma), axis=0)
        b = np.append(b, -(sum_val)*penalty_gamma)
        ###############################################
        ###############################################

        N = int(60 / time_interval_min * 24)-1
        M = len(OD_paths)

        print('A',np.max(A))
        print('b',np.max(b))
#         (q_est, r_norm) = sc.optimize.nnls(A, b)
        print('A shape', A.shape)
        (q_est, r_norm) = nnls(A,b,q_init,2000, 500, 5, True)
        print("solve q.")
        print('error, ', r_norm)
        print('q_est, ', q_est.sum())

        np.save(os.path.join('Q_vector', 'q_vector_2019_sym_large_beta'+
                             str(np.format_float_scientific(penalty_beta, exp_digits=2))+'_gamma'+ 
                             str(np.format_float_scientific(penalty_gamma, exp_digits=2))+'.npy'), q_est[:y1])

#         q_est = q_est[:y1]
        q_est_2d = q_est[:y1].reshape(N,M)

#         q_dict[cur_date_time] = q_est
#         q_dict_2d[cur_date_time] = q_est_2d[:-1,:]

#         print('q_dict_2d,', q_dict_2d[cur_date_time].shape)

        x_est =  RP.dot(q_est[:y1])

        total_flow = np.sum(A_ones.dot(q_est[:y1]))
        axis_y = i%3
        axis_x = int(i/3)

        plt.scatter(x_o, x_est, c = 'b', s=2)
        plt.plot(x_o, x_o, 'r')

        plt.title(date_str)
        plt.savefig(date_str+'.png')
        plt.close()
        date_need_to_finish.append(single_date)
        i += 1
            
#         cur_date_time = cur_date_time + delta_date
        return total_flow

penalty_beta = 10
penalty_gamma = 1

cur_date_time = datetime.datetime.combine(start_date, time_basis)
total_flow = solve_nnls(cur_date_time)
total_flow_constraints = solve_nnls(cur_date_time, total_flow, penalty_beta, penalty_gamma)

