#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random


# In[2]:



## read in location file
location_file = "../PeMS_rawData/Location/d07_text_meta_2019_11_09.txt" ## need to change read in method

loc_df = pd.read_csv(location_file, sep="\t", header = [0])


# In[3]:


## read in daily volume
vol_file = "../PeMS_rawData/district_7_2020/d07_text_station_5min_2020_03_20.txt" ## need to change read in method

colnames=['Datetime', 'Station', 'Freeway', 'Direction', 'Flow', 'Spd'] 

vol_df = pd.read_csv(vol_file, sep=",", usecols = [0,1,3,4,9,11], names=colnames, header = None)


# In[5]:


grouped = vol_df.groupby('Station')
station_dict = {}
available_stat = []
for name in vol_df['Station'].unique():
    station_dict[name] = grouped.get_group(name)
#     print(station_dict[name])
    
    percent_missing = station_dict[name].isnull().sum() * 100 / len(station_dict[name])
    
    if percent_missing['Flow'] == 0 and percent_missing['Spd'] == 0:
        available_stat.append(name)
    


# In[7]:


edge_file = "./processed_data/Edge_list_large.csv"
node_file = "./processed_data/node_list_large.csv"

edge_df = pd.read_csv(edge_file, header = [0])
node_df = pd.read_csv(node_file, header = [0])

edge_df['highway'] = edge_df.highway.str.extract(r'(\d+[.\d]*)')
edge_df['direction'] = edge_df['direction'].astype(str).str[0]


# In[8]:


volume_link_dict = {}
spd_link_dict = {}
for i in range(len(edge_df)):
#     i = 53
#     print('edge_df ', edge_df.loc[i])
    from_node_id = edge_df.loc[i,'from_node']
    to_node_id = edge_df.loc[i,'to_node']
#     print(to_node_id)
    direction = edge_df.loc[i,'direction']
    highway = int(edge_df.loc[i,'highway'])
    
#     print('direction, ', direction, ' highway, ', highway)

    lng1 = float(node_df[node_df['node_id']==from_node_id]['lng'])
    lat1 = float(node_df[node_df['node_id']==from_node_id]['lat'])
    lng2 = float(node_df[node_df['node_id']==to_node_id]['lng'])
    lat2 = float(node_df[node_df['node_id']==to_node_id]['lat'])
    eps = 0.0001
#     print('lat1, ', lat1, 'lat2, ', lat2)
    import geopy.distance

    coords_1 = (lat1, lng1)
    coords_2 = (lat1, lng2)

    length = geopy.distance.geodesic(coords_1, coords_2).miles

#     print(length)
    
    if direction == 'E'or direction == 'W':
#         print('i am here1')
        name_list = loc_df.loc[(loc_df.Longitude <= max(lng1,lng2)+eps)                                     & (loc_df.Longitude >= min(lng1,lng2)-eps)                                     & (loc_df.Fwy == highway)                                    & (loc_df.Dir == direction)]['ID'].to_list()
    else:
#         print('i am here2')
        name_list = loc_df.loc[(loc_df.Latitude <= max(lat1,lat2)+eps)                                     & (loc_df.Latitude >= min(lat1,lat2)-eps)                                     & (loc_df.Fwy == highway)                                    & (loc_df.Dir == direction)]['ID'].to_list()

#     print(type(vol_df['Station']))
    tmp_df = vol_df[vol_df['Station'].isin(name_list)]
    tmp_density = pd.DataFrame()
    tmp_spd = pd.DataFrame()
#     print('tmp_df, ', tmp_df)
    tmp_df_dict = {}
    for name in tmp_df['Station'].unique():
        tmp_df_dict[name] = station_dict[name][['Datetime','Flow','Spd']]

        percent_missing = station_dict[name].isnull().sum() * 100 / len(station_dict[name])

        if (percent_missing['Flow'] == 0) & (percent_missing['Spd'] == 0):
#             print(tmp_df_dict[name])
            tmp_density[name] = (tmp_df_dict[name]['Flow']/(tmp_df_dict[name]['Spd']/12)).to_list()
            tmp_spd[name] = tmp_df_dict[name]['Spd'] #mph
        
        date = (pd.to_datetime(tmp_df_dict[name]["Datetime"]).dt.date.astype(str)).tolist()
#         date = pd.to_datetime(tmp_df_dict[name]["Datetime"]).dt.date
#         print('date, ', date)
        time = (pd.to_datetime(tmp_df_dict[name]["Datetime"]).dt.time.astype(str)).tolist()
#         time = pd.to_datetime(tmp_df_dict[name]["Datetime"]).dt.time
#         print('time, ', time)
    tmp_density['link_flow'] = tmp_density.mean(axis=1) * length
    tmp_density["date"] = pd.to_datetime(date).date
    tmp_density["time"] = pd.to_datetime(time).time
#     print('tmp_density, ', tmp_density)
    tmp_spd['link_spd'] = tmp_spd.mean(axis=1)
    tmp_spd["date"] = pd.to_datetime(date).date
    tmp_spd["time"] = pd.to_datetime(time).time
    
    volume_link_dict[edge_df.loc[i,'link_id']] = pd.pivot_table(tmp_density, index= tmp_density.date, columns=tmp_density.time, values = 'link_flow')
#     print('index type,', type(volume_link_dict[edge_df.loc[i,'link_id']].index[0]))
#     print('col type,', type(volume_link_dict[edge_df.loc[i,'link_id']].columns[0]))
    spd_link_dict[edge_df.loc[i,'link_id']] = pd.pivot_table(tmp_spd, index= tmp_spd.date, columns=tmp_spd.time, values = 'link_spd')
#     print('index type,', type(spd_link_dict[edge_df.loc[i,'link_id']].index[0]))
#     print('col type,', type(spd_link_dict[edge_df.loc[i,'link_id']].columns[0]))
    
    df = spd_link_dict[edge_df.loc[i,'link_id']]
    if df.isnull().sum().sum() * 100 / len(df) != 0:
        print('df, ', df)
        print('nan not zero: ', edge_df.loc[i,'link_id'])


# In[8]:


print(volume_link_dict)


# In[9]:


import pickle
with open("./processed_data/20190320_large/volume_dict_link_4.pkl", 'wb') as handle:
    pickle.dump(volume_link_dict, handle, protocol=4)
with open("./processed_data/20190320_large/speed_dict_link_4.pkl", 'wb') as handle:
    pickle.dump(spd_link_dict, handle, protocol=4)



