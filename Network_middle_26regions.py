#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import pickle
import copy
from scipy.sparse import csr_matrix
from scipy import io
import seaborn as sns
import joblib
from joblib import Parallel, delayed
import random
import csv
#from mpl_toolkits.basemap import Basemap as Basemap


# In[14]:


class intersection:
    def __init__(self, lat, lng, on_street, to_street = None, from_street = None, link_type='HIGHWAY', num_lanes=-1, direction = None):
        self.link_id = link_id
        self.lat = lat
        self.lng = lng
        self.on_street = on_street
        self.to_street = to_street
        self.from_street = from_street
        self.link_type = link_type
        self.num_lanes = num_lanes
        self.direction = direction


# Todo:
# 1. Add a function to control the length of link in network
# 2. Add a function to deal with missing date
#     method 1: find the intersection of all possible date
#     method 2: use the data of nearest sensor

# In[15]:


####################################
##convert lat and lng to cartisian##
####################################
def LLHtoECEF(lat, lon, alt):
    # see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html

    rad = np.float64(6378137.0)        # Radius of the Earth (in meters)
    f = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model
    cosLat = np.cos(lat)
    sinLat = np.sin(lat)
    FF     = (1.0-f)**2
    C      = 1/np.sqrt(cosLat**2 + FF * sinLat**2)
    S      = C * FF

    x = (rad * C + alt)*cosLat * np.cos(lon)
    y = (rad * C + alt)*cosLat * np.sin(lon)
    z = (rad * S + alt)*sinLat

    return (x/1000, y/1000, z/1000)

####################################
##########calculate distance #######
####################################
def haversine(Olat,Olon, Dlat,Dlon):

    radius = 6371.  # km

    d_lat = np.radians(Dlat - Olat)
    d_lon = np.radians(Dlon - Olon)
    a = (np.sin(d_lat / 2.) * np.sin(d_lat / 2.) +
         np.cos(np.radians(Olat)) * np.cos(np.radians(Dlat)) *
         np.sin(d_lon / 2.) * np.sin(d_lon / 2.))
    c = 2. * np.arctan2(np.sqrt(a), np.sqrt(1. - a))
    d = radius * c
    return d


# In[16]:


Nodes = pd.read_pickle("./modified_location_large.pkl")
#Nodes['ID'] = np.arange(len(Nodes))
Nodes = Nodes.reset_index(drop=True)
# print(Nodes)
# print(Nodes.at[1,"link_id"])

###Input parameter###
#h_list=['SR-90','I-10','I-110','I-5','I-405','I-105','I-710']
#h_list=['I-110']
h_list = Nodes.on_street.unique()
# print(h_list)


# Build the network:
# Method 1: Add the node and add edges at the same time
# Method 2: Add nodes and then add edges
# 
# Add edges method: 
# 1. with the same on_street and direction, build the node list.
# 2. Sort nodes by their location
# 3. Connect them one by one
# 
# 1. get the list of on_street and direction (done)
# 2. Get the data for the same combination
# 
# Build the intersect
# <!-- 1. find a bunch of node that have neariest position and in different road
# 1.1 sort all nodes with elngth
# 1.2 pick selected rows
# 2. connect them based on direction
# 3. all intersection -->
# 1. For each road and direction, iterate over nodes
# 2. For each node, find nearest nodes on other road.
# 3. if latitude and longitude within certain range, record it

# Build the network algorithm:
# 1. find all intersection node.
# 2. connect the intersection nodes on the same highway
# 3. connect the intersection node with the near node on another high way
# 4. Build a list to calculate all speed and volume.
# 
# Every highway have a list of intersection node

# In[17]:


### Drop duplicated ###
Nodes = Nodes.drop_duplicates(subset = 'link_id', keep = "first")
#Nodes['link_id'] = Nodes.reset_index().index
# print("Nodes: ", Nodes)


# In[18]:


pd.options.display.max_rows = 4000
Highway_list = Nodes.groupby(['on_street','direction']).size().reset_index().rename(columns={0:'count'})
#print(Highway_list)
# print(Highway_list.loc[Highway_list['on_street'].isin(h_list)])
Highway_list = Highway_list.loc[Highway_list['on_street'].isin(h_list)]


# In[19]:


plt.close('all')
DG = nx.DiGraph()
node_id=0

#print(node_attr[2])
OD_list = pd.DataFrame(columns=['origin', 'destination'])
link_id=0;
target_length=0;

intersection_list = pd.DataFrame(columns=['node_id', 'lng', 'lat' ,'highwaylist'])
intersection_list['highwaylist'] = intersection_list['highwaylist'].astype(object)

for idx,rows in Highway_list.iterrows():
    #print(rows['on_street'])
    #print(idx)
#     print(rows['on_street'], rows['direction'])
    
    #################################
    #######Connect high way##########
    #################################
    Data = Nodes.loc[np.logical_and(Nodes.on_street == rows['on_street'], Nodes.direction == rows['direction'])]
    if rows['direction'] == 'NORTH':
        Sorted_data=Data.sort_values(by=['lat'], ascending=True)
    elif rows['direction'] == 'SOUTH':
        Sorted_data=Data.sort_values(by=['lat'], ascending=False)
    elif rows['direction'] == 'EAST':
        Sorted_data=Data.sort_values(by=['lng'], ascending=True)
    elif rows['direction'] == 'WEST':
        Sorted_data=Data.sort_values(by=['lng'], ascending=False)
    else:
        print("error!")
    Sorted_data = Sorted_data.reset_index(drop=True)
    PartNode = Nodes.loc[Nodes['on_street'].isin(h_list)]
    Notselect_data = PartNode.loc[np.logical_and(PartNode.on_street != rows['on_street'], PartNode.direction != rows['direction'])]
    ###########################################################
    ########Add the first and last to intersection list########
    ###########################################################
#     rows_next = Sorted_data.iloc[1]
#     if (rows['direction'] == 'EAST') or (rows['direction'] == 'WEST'):
#         intersection=Notselect_data.loc[(Notselect_data['lat'] > rows1['lat']-0.005) \
#                                     & (Notselect_data['lat'] < rows1['lat']+0.005) & \
#                        (Notselect_data['lng'] >= min(rows1['lng'],rows1_next['lng']))
#                                     & (Notselect_data['lng'] <= max(rows1['lng'],rows1_next['lng']))]
#     elif (rows['direction'] == 'NORTH') or (rows['direction'] == 'SOUTH'):
#         intersection=Notselect_data.loc[(Notselect_data['lng'] > rows1['lng']-0.005) 
#                                     & (Notselect_data['lng'] < rows1['lng']+0.005) & 
#                        (Notselect_data['lat'] >= min(rows1['lat'],rows1_next['lat'])) 
#                                     & (Notselect_data['lat'] <= max(rows1['lat'],rows1_next['lat']))]

#     if intersection.empty:
    df_temp3 = pd.DataFrame(columns=['node_id', 'lng', 'lat' ,'highwaylist'])
    df_temp3['highwaylist'] = df_temp3['highwaylist'].astype(object)

    if (rows['direction'] == 'EAST') or (rows['direction'] == 'WEST'): 
        highwaylist_new = [[rows['on_street'], 'EAST'],[rows['on_street'], 'WEST']]
        highwaylist_new1 = [[rows['on_street'], 'EAST'],[rows['on_street'], 'WEST']]
    elif (rows['direction'] == 'NORTH') or (rows['direction'] == 'SOUTH'):
        highwaylist_new = [[rows['on_street'], 'NORTH'],[rows['on_street'], 'SOUTH']]
        highwaylist_new1 = [[rows['on_street'], 'NORTH'],[rows['on_street'], 'SOUTH']]

    if node_id == 0:
        df_temp3.loc[0] = (node_id, Sorted_data.iloc[0]['lng'], Sorted_data.iloc[0]['lat'], highwaylist_new)
        node_id = node_id +1;
        df_temp3.loc[1] = (node_id, Sorted_data.iloc[-1]['lng'], Sorted_data.iloc[-1]['lat'], highwaylist_new1)
        node_id = node_id +1;

        intersection_list = intersection_list.append(df_temp3)
        intersection_list = intersection_list.reset_index(drop=True)
    else:
        for index3, rows3 in intersection_list.iterrows():
            if sorted(rows3['highwaylist']) == sorted(highwaylist_new):
                break;
            elif index3 == len(intersection_list)-1:
                df_temp3.loc[0] = (node_id, Sorted_data.iloc[0]['lng'], Sorted_data.iloc[0]['lat'], highwaylist_new)
                node_id = node_id +1;
                df_temp3.loc[1] = (node_id, Sorted_data.iloc[-1]['lng'], Sorted_data.iloc[-1]['lat'], highwaylist_new1)
                node_id = node_id +1;
                intersection_list = intersection_list.append(df_temp3)
                intersection_list = intersection_list.reset_index(drop=True)
#     OD_list = OD_list.append(df_temp3)
    #print("OD_list1: ", OD_list)
        
    #################################
    #######Build the intersect#######
    #################################
        #Notselect_data = PartNode.loc[np.logical_and(PartNode.on_street != 'I-10', PartNode.direction != 'EAST')]

    for index, rows1 in Sorted_data.iterrows():
        ##if north and west
#         print("rows1: ", rows1)
#         print("index1: ", index," ",len(Sorted_data))
        if index == len(Sorted_data)-1:
            break
        rows1_next = Sorted_data.iloc[index+1]
        if (rows1['direction'] == 'EAST') or (rows1['direction'] == 'WEST'):
            intersection=Notselect_data.loc[(Notselect_data['lat'] > rows1['lat']-0.005) 
                                        & (Notselect_data['lat'] < rows1['lat']+0.005) & 
                           (Notselect_data['lng'] >= min(rows1['lng'],rows1_next['lng']))
                                        & (Notselect_data['lng'] <= max(rows1['lng'],rows1_next['lng']))]
        elif (rows1['direction'] == 'NORTH') or (rows1['direction'] == 'SOUTH'):
            intersection=Notselect_data.loc[(Notselect_data['lng'] > rows1['lng']-0.01) 
                                        & (Notselect_data['lng'] < rows1['lng']+0.01) & 
                           (Notselect_data['lat'] >= min(rows1['lat'],rows1_next['lat'])) 
                                        & (Notselect_data['lat'] <= max(rows1['lat'],rows1_next['lat']))]
            
        if not intersection.empty:
            highwaylist = intersection.drop_duplicates(['on_street','direction'])[['on_street','direction']].values.tolist()
#             print("intersection1: ", intersection)
#             print("highwaylist: ", highwaylist[['on_street','direction']])
            ## Define intersection
            lng = (rows1['lng'] + rows1_next['lng'])/2
            lat = (rows1['lat'] + rows1_next['lat'])/2
            if (rows['direction'] == 'EAST') or (rows['direction'] == 'WEST'): 
                highwaylist_new = [[rows1['on_street'], 'EAST'],[rows1['on_street'], 'WEST']]
            elif (rows['direction'] == 'NORTH') or (rows['direction'] == 'SOUTH'):
                highwaylist_new = [[rows1['on_street'], 'NORTH'],[rows1['on_street'], 'SOUTH']]
                
#             print('highwaylist_new1: ', highwaylist_new)
            for i in range(len(highwaylist)):
                highwaylist_new.append(highwaylist[i])
#             print('highwaylist_new: ', highwaylist_new)
            df_temp1 = pd.DataFrame(columns=['node_id', 'lng', 'lat' ,'highwaylist'])
            df_temp1['highwaylist'] = df_temp1['highwaylist'].astype(object)
            df_temp1.loc[0] = (node_id, lng, lat, highwaylist_new)
            if node_id == 0:
                intersection_list = intersection_list.append(df_temp1)
                node_id = node_id+1
            else:
                for index2, rows2 in intersection_list.iterrows():
#                     print('index2: ', index2);
#                     print('len: ', len(intersection_list))
                    if sorted(rows2['highwaylist']) == sorted(highwaylist_new):
                        break;
                    elif index2 == len(intersection_list)-1:
                        intersection_list = intersection_list.append(df_temp1)
                        node_id = node_id+1;
                        intersection_list = intersection_list.reset_index(drop=True)
                        
intersection_list1 = intersection_list;                        
# intersection_list = intersection_list.drop([61,60,62,100,89,59,108])
# df_temp3 = pd.DataFrame(columns=['node_id', 'lng', 'lat' ,'highwaylist'])
# df_temp3['highwaylist'] = df_temp3['highwaylist'].astype(object)
# df_temp3.loc[44] = (44, -118.352, 34.0341570, [['I-10', 'EAST'], ['I-10', 'WEST']])
# df_temp3.loc[45] = (45, -118.280689, 33.981204, [['I-110', 'SOUTH'], ['I-110', 'NORTH']])
# df_temp3.loc[46] = (46, -118.167871, 33.966909, [['I-710', 'SOUTH'], ['I-710', 'NORTH']])

# intersection_list = intersection_list.append(df_temp3)
# intersection_list.loc[37]['highwaylist'].append(['I-110', 'NORTH'])
# intersection_list.loc[25]['highwaylist'].append(['I-110','NORTH'])
# intersection_list.loc[23]['highwaylist'].append(['I-710','NORTH'])
# intersection_list.loc[91]['highwaylist'].extend([['I-5', 'NORTH'], ['I-5', 'SOUTH']])
# intersection_list.loc[98]['highwaylist'].extend([['I-5', 'NORTH'], ['I-5', 'SOUTH']])
# intersection_list.loc[36]['highwaylist'].extend([['I-5', 'NORTH'], ['I-5', 'SOUTH']])


# intersection_list.loc[6]['highwaylist'].append(['I-710','SOUTH'])
# intersection_list.loc[26]['highwaylist'].extend([['I-10', 'EAST'], ['I-10', 'WEST']])

print("final, ", intersection_list)


# In[23]:


intersection_list = intersection_list1;
intersection_list = intersection_list.drop([61,35,88,90,61,60,62,100,89,46,101,94,97,104,96,93,57,103,14])
intersection_list = intersection_list.drop([105,106,82,69,124,122,49,121,43,48,22,21,68,74,72,108])
intersection_list = intersection_list.drop([0,1,2,3,4,5,6,7])
intersection_list = intersection_list.drop([63,31,55,113,117,17,119,110,118,16,111,114,112,64,79,81,54,66]) #tmp
intersection_list = intersection_list.drop([67,80,53,116,65,56,19,85]) #tmp
intersection_list = intersection_list.drop([34,32,87]) #tmp
intersection_list = intersection_list.drop([86,83,12,33,13,11,76,84]) #tmp



# intersection_list.loc[36]['highwaylist'].extend([['I-5', 'NORTH'], ['I-5', 'SOUTH']])
intersection_list.loc[91]['highwaylist'].extend([['I-5', 'NORTH'], ['I-5', 'SOUTH']])
intersection_list.loc[98]['highwaylist'].extend([['I-5', 'NORTH'], ['I-5', 'SOUTH'],['I-405', 'NORTH'],                                                  ['I-405', 'SOUTH'], ['SR-118', 'EAST'], ['SR-118', 'WEST'],                                                ['I-210', 'EAST'], ['I-210', 'WEST']])
# intersection_list.loc[42]['highwaylist'].extend([['I-5', 'NORTH'], ['I-5', 'SOUTH']])
intersection_list.loc[102]['highwaylist'].extend([['I-5', 'NORTH'], ['I-5', 'SOUTH']])
intersection_list.loc[92]['highwaylist'].extend([['SR-170', 'NORTH'], ['SR-170', 'SOUTH'],['SR-101', 'NORTH'], ['SR-101', 'SOUTH']])


intersection_list.loc[44]['highwaylist'].extend(['SR-101', 'NORTH'])
intersection_list.loc[40]['highwaylist'].append(['SR-2', 'WEST'])
intersection_list.loc[95]['highwaylist'].append(['SR-2', 'EAST'])
intersection_list.loc[39]['highwaylist'].extend([['SR-134', 'WEST']])
intersection_list.loc[44]['highwaylist'].extend([['SR-101', 'EAST'], ['SR-101', 'WEST']])


intersection_list.loc[77]['highwaylist'].extend([['I-405', 'NORTH']])
intersection_list.loc[41]['highwaylist'].extend([['I-605', 'NORTH'], ['I-605', 'SOUTH']])
intersection_list.loc[29]['highwaylist'].extend([['I-405', 'NORTH']])
intersection_list.loc[52]['highwaylist'].extend([['I-605', 'NORTH']])
intersection_list.loc[70]['highwaylist'].extend([['I-210', 'EAST'], ['I-210', 'WEST']])
intersection_list.loc[18]['highwaylist'].extend([['I-210', 'EAST'], ['I-210', 'WEST'],                                                 ['SR-60', 'EAST'],['SR-60', 'EAST']])
intersection_list.loc[8]['highwaylist'].extend([])

df_temp3 = pd.DataFrame(columns=['node_id', 'lng', 'lat' ,'highwaylist'])
df_temp3['highwaylist'] = df_temp3['highwaylist'].astype(object)
df_temp3.loc[131] = (131, -118.25178, 34.10304, [['SR-2', 'EAST'], ['SR-2', 'WEST'], ['I-5', 'NORTH'], ['I-5', 'SOUTH']])
df_temp3.loc[132] = (132, -118.22279, 34.08119, [['I-110', 'SOUTH'], ['I-110', 'NORTH'],['I-5', 'NORTH'], ['I-5', 'SOUTH']])
df_temp3.loc[133] = (133, -118.250, 34.06328, [['SR-101', 'NORTH'], ['SR-101', 'SOUTH'],['I-110', 'SOUTH'], ['I-110', 'NORTH']])
df_temp3.loc[134] = (134, -118.273, 34.03747, [['I-110', 'SOUTH'], ['I-110', 'NORTH'],['I-10', 'EAST'], ['I-10', 'WEST']])
df_temp3.loc[135] = (135, -118.21760, 34.05539, [['I-5', 'NORTH'], ['I-5', 'SOUTH'],['I-10', 'EAST'], ['I-10', 'WEST'],['SR-101', 'NORTH'], ['SR-101', 'SOUTH']])
df_temp3.loc[136] = (136, -118.219, 34.030, [['SR-60', 'EAST'], ['SR-60', 'WEST'],['I-5', 'NORTH'], ['I-5', 'SOUTH'],['I-10', 'EAST'], ['I-10', 'WEST'],['SR-101', 'NORTH'], ['SR-101', 'SOUTH']])
df_temp3.loc[137] = (137, -118.17218, 34.015, [['I-5', 'NORTH'], ['I-5', 'SOUTH'],['I-710', 'SOUTH'], ['I-710', 'NORTH']])
df_temp3.loc[138] = (138, -118.159, 34.03425, [['SR-60', 'EAST'], ['SR-60', 'WEST'],['I-710', 'SOUTH'], ['I-710', 'NORTH']])
df_temp3.loc[139] = (139, -118.16570, 34.06077, [['I-10', 'EAST'], ['I-10', 'WEST'],['I-710', 'SOUTH'], ['I-710', 'NORTH']])
# df_temp3.loc[140] = (140, -118.60429, 34.27692, [['SR-118', 'EAST'], ['SR-118', 'WEST']])
# df_temp3.loc[141] = (141, -118.63370, 34.26852, [['SR-118', 'EAST'], ['SR-118', 'WEST']])
# df_temp3.loc[142] = (142, -118.60429, 34.16875, [['SR-101', 'EAST'], ['SR-101', 'WEST']])
# df_temp3.loc[143] = (143, -118.68989, 34.14879, [['SR-101', 'EAST'], ['SR-101', 'WEST']])
# df_temp3.loc[144] = (144, -118.81171, 34.15194, [['SR-101', 'EAST'], ['SR-101', 'WEST']])
df_temp3.loc[145] = (145, -118.3299, 34.46583, [['SR-14', 'NORTH'], ['SR-14', 'SOUTH']])
# df_temp3.loc[146] = (146, -118.69258, 34.40545, [['SR-126', 'EAST'], ['SR-126', 'WEST']])
# df_temp3.loc[147] = (147, -118.07373, 34.14781, [['I-210', 'EAST'], ['I-210', 'WEST']])
# df_temp3.loc[148] = (148, -118.07294, 34.07114, [['I-10', 'EAST'], ['I-10', 'WEST']])
# df_temp3.loc[61] = (61, -118.31167, 34.47213, [['SR-14', 'NORTH'], ['SR-14', 'SOUTH']])

df_temp3.loc[149] = (149, -118.35158, 33.88295, [['I-405', 'NORTH'], ['I-405', 'SOUTH']])
# df_temp3.loc[150] = (148, -118.07294, 34.07114, [['I-10', 'EAST'], ['I-10', 'WEST']])
# df_temp3.loc[151] = (148, -118.07294, 34.07114, [['I-10', 'EAST'], ['I-10', 'WEST']])
# df_temp3.loc[152] = (148, -118.07294, 34.07114, [['I-10', 'EAST'], ['I-10', 'WEST']])

# medium network
df_temp3.loc[150] = (150, -118.279, 33.981, [['I-110', 'SOUTH'], ['I-110', 'NORTH']])
df_temp3.loc[151] = (151, -118.469, 34.075, [['I-405', 'NORTH'], ['I-405', 'SOUTH']])

#
# add local road node
### region 4
# #### local
# df_temp3.loc[152] = (152, -118.35064, 34.09017, [['SantaMonicaBLVD', 'WEST'], ['SantaMonicaBLVD', 'EAST']])
# df_temp3.loc[153] = (153, -118.309083, 34.06092, [['WesternAve4', 'NORTH'], ['WesternAve4', 'SOUTH']])
# #### highway
# df_temp3.loc[154] = (154, -118.307949, 34.092664, [['SantaMonicaBLVD', 'WEST'], ['SantaMonicaBLVD', 'EAST'],\
#                                                   ['WesternAve4', 'NORTH'], ['WesternAve4', 'SOUTH'],\
#                                                   ['SR-101', 'NORTH'], ['SR-101', 'SOUTH']])

# ### region 6
# #### local
# df_temp3.loc[155] = (155, -118.30905, 33.99195, [['WesternAve6', 'NORTH'], ['WesternAve6', 'SOUTH']])
# #### highway
# df_temp3.loc[156] = (154, -118.30894, 34.03675, [['WesternAve6', 'NORTH'], ['WesternAve6', 'SOUTH'],\
#                                                   ['I-10', 'EAST'], ['I-10', 'WEST']])

# add new nodes to cover all areas
df_temp3.loc[152] = (152, -118.325405, 34.104729, [['SR-101', 'NORTH'], ['SR-101', 'SOUTH']])
df_temp3.loc[153] = (153, -118.350162, 34.034160, [['I-10', 'EAST'], ['I-10', 'WEST']])
df_temp3.loc[154] = (154, -118.170455, 33.956619, [['I-710', 'SOUTH'], ['I-710', 'NORTH']])

intersection_list = intersection_list.append(df_temp3)

intersection_list = intersection_list.drop([75,50,123,107,37,9,115,149,120,20,                                            42,45,59,36,135,38,109,91,145]) #new


df_temp4 = pd.DataFrame(columns=['on_street', 'direction', 'count'])
df_temp4.loc[50] = ('SR-101', 'EAST',3)
df_temp4.loc[51] = ('SR-101', 'WEST',3)
Highway_list = Highway_list.append(df_temp4)
print(Highway_list)
# Highway_list.extend([['SR-126', 'EAST'], ['SR-126', 'WEST']])
intersection_list = intersection_list.reset_index(drop=True)
intersection_list['node_id'] = intersection_list.index
print("final, ", intersection_list)
print("len, ", len(intersection_list.index))


# In[24]:


intersection_df = pd.DataFrame(columns=['node_id', 'lng', 'lat' ,'highway','direction'])
for index, row in intersection_list.iterrows():
    for i in range(len(row['highwaylist'])):
        df_temp = pd.DataFrame(columns=['node_id', 'lng', 'lat' ,'highway','direction'])
        df_temp.loc[0] = (row['node_id'],row['lng'],row['lat'],row['highwaylist'][i][0], row['highwaylist'][i][1])
        intersection_df = intersection_df.append(df_temp)
intersection_df = intersection_df.reset_index(drop=True)

####### Connect the nextwork ########
# Highway_list = Nodes.groupby(['on_street','direction']).size().reset_index().rename(columns={0:'count'})
# Highway_list = Highway_list.loc[Highway_list['on_street'].isin(h_list)]
Edge_list = pd.DataFrame(columns=['from_node', 'to_node', 'link_id' ,'length'])
link_id = 0;
for idx,rows in Highway_list.iterrows():
    
    #################################
    #######Connect high way##########
    #################################
    Data = intersection_df.loc[np.logical_and(intersection_df.highway == rows['on_street'], 
                                              intersection_df.direction == rows['direction'])]
    if rows['direction'] == 'NORTH':
        Sorted_data=Data.sort_values(by=['lat'], ascending=True)
    elif rows['direction'] == 'SOUTH':
        Sorted_data=Data.sort_values(by=['lat'], ascending=False)
    elif rows['direction'] == 'EAST':
        Sorted_data=Data.sort_values(by=['lng'], ascending=True)
    elif rows['direction'] == 'WEST':
        Sorted_data=Data.sort_values(by=['lng'], ascending=False)
    else:
        print("error!")
    
    for i in range(len(Sorted_data)-1):
#         print("i: ", i)
#         print("len: ", len(Sorted_data))
        #print(Sorted_data.iloc[i-1]['ID'])
        df_temp = pd.DataFrame(columns=['from_node', 'to_node', 'link_id' ,'length','highway','direction'])
        j = i + 1
        length = haversine(Sorted_data.iloc[i]['lat'],Sorted_data.iloc[i]['lng'],                                    Sorted_data.iloc[j]['lat'],Sorted_data.iloc[j]['lng'])
#         while (length < 1) & (j < len(Sorted_data)-1):
#             j = j+1
#             length = haversine(Sorted_data.iloc[i]['lat'],Sorted_data.iloc[i]['lng'],\
#                                     Sorted_data.iloc[j]['lat'],Sorted_data.iloc[j]['lng'])
        df_temp.loc[0] = (Sorted_data.iloc[i]['node_id'],Sorted_data.iloc[j]['node_id'], 
                          link_id,length,rows['on_street'],rows['direction'])
        link_id = 1 + link_id;
        Edge_list = Edge_list.append(df_temp)
        i = j
        
Edge_list = Edge_list[Edge_list['length'] > 0] 
print(Edge_list)

# df_temp = pd.DataFrame(columns=['from_node', 'to_node', 'link_id' ,'length','highway','direction'])
# length = haversine(intersection_list.iloc[34]['lat'],intersection_list.iloc[34]['lng'],\
#                             intersection_list.iloc[24]['lat'],intersection_list.iloc[24]['lng'])
# df_temp.loc[0] = (34,24, 
#                   link_id,length,'I-10','EAST')
# link_id = 1 + link_id;
# df_temp.loc[1] = (24,34, 
#                   link_id,length,'I-10','WEST')
# link_id = 1 + link_id;
# length = haversine(intersection_list.iloc[7]['lat'],intersection_list.iloc[7]['lng'],\
#                             intersection_list.iloc[26]['lat'],intersection_list.iloc[26]['lng'])
# df_temp.loc[3] = (26,7, 
#                   link_id,length,'I-5','NORTH')
# link_id = 1 + link_id;
# df_temp.loc[4] = (7,26, 
#                   link_id,length,'I-5','SOUTH')
# link_id = 1 + link_id;
# Edge_list = Edge_list.append(df_temp)

# Edge_list = Edge_list.reset_index()

# Edge_list = Edge_list.drop(Edge_list.index[100])


# In[28]:


Edge_list['weight']=1
DG = nx.from_pandas_edgelist(Edge_list, source='from_node', target='to_node', edge_attr=['link_id','length','weight'], create_using=nx.DiGraph())

intersection_list['pos'] = intersection_list[['lng', 'lat']].apply(tuple, axis=1)
node_attr = intersection_list.set_index('node_id').to_dict('index')

nx.set_node_attributes(DG, node_attr)
pos=nx.get_node_attributes(DG,'pos')
plt.rcParams['figure.figsize'] = [20, 20]
nx.draw_networkx(DG,pos,node_size=2,node_color='blue',with_labels=True, font_size=20, horizontalalignment = 'left',verticalalignment='bottom',arrows=True, arrowstyle="->",arrowsize=10)


# In[29]:


##########################################
#######Calculate average vol&spd##########
##########################################
with open('volume_dict_new.pickle', 'rb') as handle:
    volume_dict_new = pickle.load(handle)
with open('speed_dict_new.pickle', 'rb') as handle:
    speed_dict_new = pickle.load(handle)
    
link_highwaylist = Edge_list.drop_duplicates(['highway','direction'])[['highway','direction']]
print(link_highwaylist)
volume_dict_link = {}
speed_dict_link = {}
eps=0.01
Edge_list= Edge_list.reset_index(drop=True)
x = pd.DataFrame()
y = pd.DataFrame()
for index, row in Edge_list.iterrows():
#     print('index:, ',index)
    Data = Nodes.loc[np.logical_and(Nodes.on_street == row['highway'], Nodes.direction == row['direction'])]
    if row['direction'] == 'NORTH':
        Sorted_data=Data.sort_values(by=['lat'], ascending=True)
        lat1 = intersection_list[intersection_list['node_id']==row['to_node']]['lat'].values[0]
        lat2 = intersection_list[intersection_list['node_id']==row['from_node']]['lat'].values[0]
        name_list = Sorted_data.loc[(Sorted_data.lat <= max(lat1,lat2)+eps)                                     & (Sorted_data.lat >= min(lat1,lat2)-eps),'link_id'].values.tolist()
    elif row['direction'] == 'SOUTH':
        Sorted_data=Data.sort_values(by=['lat'], ascending=False)
        lat1 = intersection_list[intersection_list['node_id']==row['to_node']]['lat'].values[0]
        lat2 = intersection_list[intersection_list['node_id']==row['from_node']]['lat'].values[0]
        name_list = Sorted_data.loc[(Sorted_data.lat <= max(lat1,lat2)+eps)                                     & (Sorted_data.lat >= min(lat1,lat2)-eps),'link_id'].values.tolist()
    elif row['direction'] == 'EAST':
        if row['highway'] == 'SR-101':
            Data = Nodes.loc[np.logical_and(Nodes.on_street == row['highway'], Nodes.direction == 'SOUTH')]
            Sorted_data=Data.sort_values(by=['lng'], ascending=True)
            lng1 = intersection_list[intersection_list['node_id']==row['to_node']]['lng'].values[0]
            lng2 = intersection_list[intersection_list['node_id']==row['from_node']]['lng'].values[0]
            name_list = Sorted_data.loc[(Sorted_data.lng <= max(lng1,lng2)+eps)                                         & (Sorted_data.lng >= min(lng1,lng2)-eps),'link_id'].values.tolist()
        else:
            Sorted_data=Data.sort_values(by=['lng'], ascending=True)
            lng1 = intersection_list[intersection_list['node_id']==row['to_node']]['lng'].values[0]
            lng2 = intersection_list[intersection_list['node_id']==row['from_node']]['lng'].values[0]
            name_list = Sorted_data.loc[(Sorted_data.lng <= max(lng1,lng2)+eps)                                         & (Sorted_data.lng >= min(lng1,lng2)-eps),'link_id'].values.tolist()
    elif row['direction'] == 'WEST':
        if row['highway'] == 'SR-101':
            Data = Nodes.loc[np.logical_and(Nodes.on_street == row['highway'], Nodes.direction == 'NORTH')]
            Sorted_data=Data.sort_values(by=['lng'], ascending=False)
            lng1 = intersection_list[intersection_list['node_id']==row['to_node']]['lng'].values[0]
            lng2 = intersection_list[intersection_list['node_id']==row['from_node']]['lng'].values[0]
            name_list = Sorted_data.loc[(Sorted_data.lng <= max(lng1,lng2)+eps)                                         & (Sorted_data.lng >= min(lng1,lng2)-eps),'link_id'].values.tolist()
        else:
            Sorted_data=Data.sort_values(by=['lng'], ascending=False)
            lng1 = intersection_list[intersection_list['node_id']==row['to_node']]['lng'].values[0]
            lng2 = intersection_list[intersection_list['node_id']==row['from_node']]['lng'].values[0]
            name_list = Sorted_data.loc[(Sorted_data.lng <= max(lng1,lng2)+eps)                                         & (Sorted_data.lng >= min(lng1,lng2)-eps),'link_id'].values.tolist()
    else:
        print("error!")
    
#     print(name_list)
    
    vol_list=[]
    speed_list = []
    ## set sample x and y ##
    for i in range(len(name_list)):
        print(len(x.index))
        if name_list[i] in volume_dict_new:
            vol_list.append(name_list[i])
#             if (len(x.index) == 0) & (len(volume_dict_new[name_list[i]]) == 31) \
#             & (len(volume_dict_new[name_list[i]].columns) == 144):
            x = volume_dict_new[name_list[i]];
        if name_list[i] in speed_dict_new:
            speed_list.append(name_list[i])
#             if (len(y.index) == 0) & (len(speed_dict_new[name_list[i]]) == 31) \
#             & (len(speed_dict_new[name_list[i]].columns) == 144):
            y = speed_dict_new[name_list[i]];
#         print("vol_list, ", vol_list)
    for col in x.columns:
        x[col].values[:] = 0
    for i in range(len(vol_list)):
            x = x + volume_dict_new[vol_list[i]]
            x = x.replace(0.0, np.nan)
            x = x.interpolate(method='linear', axis=0)
            x = x.interpolate(method='linear', axis=1)
            x = x.fillna(value = x.mean().mean())
            
    for col in y.columns:
        y[col].values[:] = 0
    for i in range(len(speed_list)):
            y = y + speed_dict_new[speed_list[i]]
            y = y.replace(0.0, np.nan)
            y = y.interpolate(method='linear', axis=0)
            y = y.interpolate(method='linear', axis=1)
            y = y.fillna(value = y.mean().mean())
#     print("num: ", len(x.index))
    if (x.isnull().values.any() == False) & (len(x.index) != 0):
        volume_dict_link[row['link_id']] = x/len(vol_list)
    if (y.isnull().values.any() == False) & (len(y.index) != 0):
        speed_dict_link[row['link_id']] = y/len(speed_list)
    
    ##########################################
    ######## Calculate link_length ###########
    ##########################################
    length = 0
    for i in range(len(name_list)-1):
        j = i + 1
        first_node = Sorted_data.loc[Sorted_data['link_id'] == name_list[i]]
        second_node = Sorted_data.loc[Sorted_data['link_id'] == name_list[j]]
        
        length = length + haversine(first_node['lat'].values[0],first_node['lng'].values[0],                                    second_node['lat'].values[0],second_node['lng'].values[0])
#     print(length)
    Edge_list.at[index,'weight'] = length
    
print("len: ", len(speed_dict_link))
print("len: ", len(volume_dict_link))


# In[26]:


##########################################
####### Build OD list ####################
##########################################
od_list1 = []
od_list1.append(intersection_list['node_id'].tolist())
od_list1.append(intersection_list['node_id'].tolist())
# od_list1.append(Nodes['link_id'].unique().tolist())
# od_list1.append(Nodes['link_id'].unique().tolist())
print(od_list1)

file_name = "od_list.pickle"
open_file = open(file_name, "wb")
pickle.dump(od_list1, open_file)
open_file.close()


# In[27]:


with open('volume_dict_link.pickle', 'wb') as handle:
    pickle.dump(volume_dict_link, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('speed_dict_link.pickle', 'wb') as handle:
    pickle.dump(speed_dict_link, handle, protocol=pickle.HIGHEST_PROTOCOL)
nx.write_gpickle(DG,'graph.pickle')

intersection_list.to_csv(r'node_list.csv',index=False)


# In[42]:


print(Edge_list)
volume_dict_link[1]


# In[43]:


nx.write_gpickle(DG,'graph_middle.pickle')

intersection_list.to_csv(r'node_list_middle.csv',index=False)


# 1. check if it contains link between those location in that highway

# In[16]:


########## Test code ##########
eps = 0.001
rslt = Nodes.loc[np.logical_and(Nodes.on_street == 'I-405', Nodes.direction == 'NORTH')]
data = rslt.loc[(rslt.lat <= 34.366479+eps) & (rslt.lat >= 34.157897-eps),'link_id'].values.tolist()
## 15: -118.469035  34.157897
## 27: -118.502584  34.366479
print(data)

for i in data:
    if i in speed_dict_new:
        print(i)
        print(speed_dict_new[i])


# In[17]:


empty_edge_list = []
for name in volume_dict_link.keys():
    mys = volume_dict_link[name]
#     print(mys)
#     mys = mys.join(pd.DataFrame(mys.pop('volume').values.tolist()))
#     print("type, ", type(mys))
    if mys.dropna().empty == True:
#         print("Edge ID, ", name)
        empty_edge_list.append(name)
        rslt_df = Edge_list.loc[Edge_list['link_id'] == name]
        print('rslt_df, ', rslt_df)

#     print(name,", ",volume_dict_link[name])
print("len: ", len(empty_edge_list))


# In[30]:


intersection_list.to_csv(r'node_list.csv',index=False)


# In[ ]:




