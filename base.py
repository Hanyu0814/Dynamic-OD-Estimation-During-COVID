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

class Link:
    def __init__(self, ID, length):
        self.ID = ID
        self.length = length

class Path:
    def __init__(self):
        self.node_list = None
        self.link_list = None
        self.cost = 0
        self.p = None
        self.q = None
        return
    def node_to_list(self, G, link_dict):
        if self.node_list == None:
            print("Nothing to convert")
            return
        tmp = list()
        self.cost = 0
        for i in range(len(self.node_list) - 1):
           # try:
#                 print("I am here!")
                link_ID = G[self.node_list[i]][self.node_list[i+1]]["link_id"]
                
                if link_ID not in list(link_dict.keys()):
                    tmp_link = Link(link_ID, G[self.node_list[i]][self.node_list[i+1]]["length"])
                    tmp.append(tmp_link)
                    link_dict[link_ID] = tmp_link
                else:
                    tmp.append(link_dict[link_ID])
                    
                self.cost += G[self.node_list[i]][self.node_list[i+1]]["length"]
#             except:
#                 print("ERROR")
#                 print(self.node_list[i], self.node_list[i+1])
        self.link_list = tmp

def overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))

def get_finish_time(spd, length_togo, start_time, tmp_date):
    basis = datetime.datetime.combine(tmp_date, datetime.time(0,0,0))
    
    time_seq = [(datetime.datetime.combine(tmp_date, x) - basis).total_seconds() for x in spd.index]
    data = np.array(spd.tolist()).astype(np.float)
#     print data
#     print time_seq
    cur_spd = np.interp((datetime.datetime.combine(tmp_date, start_time) - basis).total_seconds(), time_seq, data) / 1600.0 * 3600.0
    try:
        new_start_time = (datetime.datetime.combine(tmp_date, start_time) + datetime.timedelta(seconds = length_togo/cur_spd)).time()
#     print "need:", length_togo/cur_spd
    except:
        new_start_time = (datetime.datetime.combine(tmp_date, start_time) + datetime.timedelta(seconds = 10)).time()
    return new_start_time


########################
##   deprecated
########################

# def get_arrival_time(start_time, link_list, spd_data, tmp_date, link_dict, spd=None):
#     if len(link_list) == 0:
#         return start_time
#     link_to_pass = link_list[0]
#     if link_to_pass.length == np.float(0):
#         link_list.pop(0)
#         return get_arrival_time(start_time, link_list, spd_data, tmp_date, link_dict)
#     if link_to_pass.ID not in spd_data.keys():
#         link_list.pop(0)
#         new_start_time = (datetime.datetime.combine(tmp_date, start_time) + datetime.timedelta(seconds = np.round(link_to_pass.fft))).time()
#         return get_arrival_time(new_start_time, link_list, spd_data, tmp_date, link_dict)
#     if type(spd) == type(None):
#         spd = spd_data[link_to_pass.ID].loc[tmp_date]
#     length_togo = link_to_pass.length
#     new_start_time = get_finish_time(spd, length_togo, start_time, tmp_date)
#     link_list.pop(0)
#     return get_arrival_time(new_start_time, link_list, spd_data, tmp_date, link_dict, spd)

# def get_ratio(path, link, h, spd_data, analysis_start_time, time_interval, tmp_date, link_dict):
#     start_time = (datetime.datetime.combine(tmp_date, analysis_start_time) + h * time_interval).time()
#     start_time2 = (datetime.datetime.combine(tmp_date, analysis_start_time) + (h+1) * time_interval).time()
#     tmp_link_list = list()
#     for tmp_link in path.link_list:
#         if link != tmp_link:
#             tmp_link_list.append(tmp_link)
#         else:
#             break
# #     print tmp_link_list
#     arrival_time = get_arrival_time(start_time, copy.copy(tmp_link_list), spd_data, tmp_date, link_dict)
#     arrival_time2 = get_arrival_time(start_time2, copy.copy(tmp_link_list), spd_data, tmp_date, link_dict)
#     p_v = get_pv(arrival_time, arrival_time2, start_time, time_interval, tmp_date)
#     if (len(p_v) > 2):
#         print start_time, arrival_time, arrival_time2
#         print p_v
#     return p_v

# row_list = list()
# col_list = list()
# data_list = list()
# for k, path in enumerate(path_list):
#     print k, len(path.link_list)
#     for a, link in enumerate(link_list):
#         if (delta[a, k] == 1):
#             for h in xrange(N):
#                 p_v = get_ratio(path, link, h, spd_data, analysis_start_time, time_interval, tmp_date, link_dict)
#                 for idx, p in enumerate(p_v):
#                     if (h + idx < N):
#                         x_loc = a + num_link * (h + idx)
#                         y_loc = k + num_path * h
#                         row_list.append(x_loc)
#                         col_list.append(y_loc)
#                         data_list.append(p)

def get_pv(arrival_time, arrival_time2, analysis_start_time, time_interval, tmp_date):
    basis = datetime.datetime.combine(tmp_date, datetime.time(0,0,0))
    arrival_time_date = datetime.datetime.combine(tmp_date, arrival_time)
    arrival_time_date2 = datetime.datetime.combine(tmp_date, arrival_time2)
    total = np.float((arrival_time_date2 -arrival_time_date).total_seconds())
    cur_time_date = datetime.datetime.combine(tmp_date, analysis_start_time)
    pv = list()
    while(cur_time_date < arrival_time_date2):
        cur_time_date2 = cur_time_date + time_interval
        overlap_zone = overlap((cur_time_date - basis).total_seconds(), (cur_time_date2 - basis).total_seconds(), (arrival_time_date - basis).total_seconds(), (arrival_time_date2 - basis).total_seconds())
#         print np.float(overlap_zone) / total
        pv.append(np.float(overlap_zone) / total)
        cur_time_date = cur_time_date2
    return pv

def get_arrival_time(start_time, link, spd_data, tmp_date, link_dict):
    link_to_pass = link
    #print('link_to_pass: ', link_to_pass)
    if link_to_pass.length == np.float(0):
        return start_time
    if link_to_pass.ID not in list(spd_data.keys()):
        #new_start_time = (datetime.datetime.combine(tmp_date, start_time) + datetime.timedelta(seconds = link_to_pass.fft)).time()
        new_start_time = (datetime.datetime.combine(tmp_date, start_time)).time()
        return new_start_time
    try:
        #print('ID: ', link_to_pass.ID)
        #print('tmp_data: ', tmp_date)
        spd = spd_data[link_to_pass.ID].loc[tmp_date]
    except:
        print("start_time, ", start_time)
        print("Except, not spd data")
        #new_start_time = (datetime.datetime.combine(tmp_date, start_time) + datetime.timedelta(seconds = link_to_pass.fft)).time()
        new_start_time = (datetime.datetime.combine(tmp_date, start_time)).time()
        return new_start_time
    length_togo = link_to_pass.length
    new_start_time = get_finish_time(spd, length_togo, start_time, tmp_date)
    return new_start_time

def get_ratio(path, h, spd_data, analysis_start_time, time_interval, tmp_date, link_dict):
    pv_dict = dict()
    start_time = (datetime.datetime.combine(tmp_date, analysis_start_time) + h * time_interval).time()
    start_time2 = (datetime.datetime.combine(tmp_date, analysis_start_time) + (h+1) * time_interval).time()
    arrival_time = copy.copy(start_time)
    arrival_time2 = copy.copy(start_time2)
    for link in path.link_list:
        arrival_time = get_arrival_time(arrival_time, link, spd_data, tmp_date, link_dict)
        arrival_time2 = get_arrival_time(arrival_time2, link, spd_data, tmp_date, link_dict)
        p_v = get_pv(arrival_time, arrival_time2, start_time, time_interval, tmp_date)
        pv_dict[link] = p_v
    return pv_dict

def get_assign_matrix(N, spd_data, analysis_start_time, time_interval, tmp_date, link_dict, link_list, link_loc, path_list):
    num_link = len(link_list)
    num_path = len(path_list)
    row_list = list()
    col_list = list()
    data_list = list()
    for k, path in enumerate(path_list):
        #if k % 1 == 0:
            #print('k: ', k, len(path_list), len(path.link_list))
        for h in range(N):
            pv_dict = get_ratio(path, h, spd_data, analysis_start_time, time_interval, tmp_date, link_dict)
            #print(pv_dict)
            for link, p_v in pv_dict.items():
                a = link_loc[link]
                print('a, ', a)
                for idx, p in enumerate(p_v):
                    y_loc = a + num_link * (h + idx)
                    x_loc = k + num_path * h
                    # print("y_loc, ", y_loc, "x_loc, ", x_loc)
                    if (h + idx < N):
                        y_loc = a + num_link * (h + idx)
                        x_loc = k + num_path * h
                        row_list.append(y_loc)
                        col_list.append(x_loc)
                        data_list.append(p)
                   
    #print("r: ", data_list, row_list, col_list)
    r = csr_matrix((data_list, (row_list, col_list)), shape=(num_link * N, num_path * N))
    return r

def save_r(N, spd_data, analysis_start_time, time_interval, single_date, link_dict, link_list, link_loc, path_list):
    #import joblib
    date_str = single_date.strftime("%Y-%m-%d")
#     print(date_str)
    r = get_assign_matrix(N, spd_data, analysis_start_time, time_interval, single_date, link_dict, link_list, link_loc, path_list)
#     print("save_r: ", r)
    joblib.dump(r, os.path.join('R_matrix', date_str+".pickle"))
    a = joblib.load(os.path.join('R_matrix', date_str+".pickle"))
#     print("loaded r:", a)
def softmax(x, theta=1):
#     print x
    """Compute softmax values for each sets of scores in x."""
    y = np.copy(x) * theta
    print(y)
    p = np.minimum(np.maximum(np.exp(y), 1e-20), 1e20) / np.sum(np.minimum(np.maximum(np.exp(y), 1e-20), 1e20), axis=0)
#     print p
    if np.isnan(p).any():
        p = np.ones(len(x)) / len(x)
    return p

def get_full_arrival_time(start_time, link_list, spd_data, tmp_date, link_dict, spd=None):
#     if len(link_list) == 0:
#         return start_time
#     link_to_pass = link_list[0]
#     if link_to_pass.length == np.float(0):
#         link_list.pop(0)
#         return get_full_arrival_time(start_time, link_list, spd_data, tmp_date, link_dict)
#     if link_to_pass.ID not in spd_data.keys():
#         link_list.pop(0)
#         new_start_time = (datetime.datetime.combine(tmp_date, start_time) + datetime.timedelta(seconds = np.round(link_to_pass.fft))).time()
#         return get_full_arrival_time(new_start_time, link_list, spd_data, tmp_date, link_dict)
#     if type(spd) == type(None):
#         spd = spd_data[link_to_pass.ID].loc[tmp_date]
#     length_togo = link_to_pass.length
#     new_start_time = get_finish_time(spd, length_togo, start_time, tmp_date)
#     link_list.pop(0)
    arrival_time = copy.copy(start_time)
    for link in link_list:
        arrival_time = get_arrival_time(arrival_time, link, spd_data, tmp_date, link_dict)
    return arrival_time

# tmp_date = datetime.date(2014, 1, 1)
def get_P(N, spd_data, analysis_start_time, time_interval, tmp_date, path_list, OD_paths):
    num_path_v = [len(x) for x in OD_paths.values()]
    num_path = np.sum(num_path_v)
    OD_list = list(OD_paths.keys())
    num_OD = len(OD_list)
    row_list = list()
    col_list = list()
    data_list = list()
    for h in range(N):
#         print h, N
        start_time = (datetime.datetime.combine(tmp_date, analysis_start_time) + h * time_interval).time()
        rs = 0
        k = 0
        for (O,D), paths in OD_paths.items():
    #         print (O,D)
            cost_list = list()
            
            for path in paths:
                arrival_time = get_full_arrival_time(start_time, path.link_list, spd_data, tmp_date, None)
                cost = (datetime.datetime.combine(tmp_date, arrival_time) - datetime.datetime.combine(tmp_date, start_time)).total_seconds()
                path.cost = cost
                cost_list.append(cost)
#                 if k < np.sum(num_path_v[0:rs+1]) and k >= np.sum(num_path_v[0:rs]):
                x_loc = h * num_OD + rs
                y_loc = h * num_path + k
                row_list.append(y_loc)
                col_list.append(x_loc)
                
                k+=1
            p_list = softmax(cost_list)
#             print('len(row_list) ', len(row_list))
#             print('len(col_list) ', len(col_list))
#             print('len(row_list)1 ', len(row_list))
#             print('len(col_list)1 ', len(col_list))
            data_list.extend(p_list)
#             print('len(data_list)1 ', len(data_list[0]), type(data_list))
            rs+=1
#             print(cost_list, p_list)
#             for idx, path in enumerate(paths):
#                 path.p = p_list[idx]
#                 data = path.p
#                 data_list.append(data)
    #         print p_list
#         for rs, (O,D) in enumerate(OD_list):
#             for k, path in enumerate(path_list):
#                 if k < np.sum(num_path_v[0:rs+1]) and k >= np.sum(num_path_v[0:rs]):
#                     x_loc = h * num_OD + rs
#                     y_loc = h * num_path + k
#                     data = path.p
#                     row_list.append(y_loc)
#                     col_list.append(x_loc)
#                     data_list.append(data)
    P = csr_matrix((data_list, (row_list, col_list)), shape=(num_path * N, num_OD * N))
    return P

def save_p(N, spd_data, analysis_start_time, time_interval, single_date, path_list, OD_paths):
    import joblib
    date_str = single_date.strftime("%Y-%m-%d")
    print('save_p', date_str)
    P = get_P(N, spd_data, analysis_start_time, time_interval, single_date, path_list, OD_paths)
    joblib.dump(P, os.path.join('P_matrix', date_str+".pickle"))


def to_south(xxx_todo_changeme):
    (O,D) = xxx_todo_changeme
    real_O = O % 1000
    real_D = D % 1000
    return real_O < real_D
