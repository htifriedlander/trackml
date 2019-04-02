
# coding: utf-8

# ## Functions necessary for pre-processing
# 
# 
# 
# use get_mappings() to see all unique {particles:superstrips} mappings for all training samples (first 100 events of total training data)
# <br>
# 
# use calc_mappings() to generate all unqiue {particles:superstrips} mappings for all training samples + writes them to file 
# 

# In[13]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from trackml.dataset import load_event, load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event
import timeit
from multiprocessing import Pool
import multiprocessing
import os


# In[23]:


def process_single_event(event_number):
    start = time.time()
    file_name = 'event00000' + str(event_number)
    event_id = file_name
    hits_orig, cells, particles, truth = load_event('data/train_sample/'+event_id)
    merge_by_hit_ids = hits_orig.merge(truth, how = 'inner', on = 'hit_id')
    merge_by_particle_ids = merge_by_hit_ids.merge(particles, how = 'inner', on = 'particle_id')
    partid_dict = {}
    hitloc_dict = {}
    for row in merge_by_particle_ids.itertuples():
        particleID = row.__getattribute__('particle_id')
        volID = row.__getattribute__('volume_id')
        layerID = row.__getattribute__('layer_id') 
        modID = row.__getattribute__('module_id')
        hitID = row.__getattribute__('hit_id')
        key_name = event_id + '-' + str(particleID)
        hitloc_dict = {'hit_id':hitID, 'volume_id':volID, 'layer_id':layerID, 'module_id': modID}
        if key_name in partid_dict:
            partid_dict[key_name].append(hitloc_dict)
        else:
            partid_dict[key_name] = [hitloc_dict]
    
    with open('mappings/'+event_id + '.json','w') as outfile:
        json.dump(partid_dict, outfile)
    
    end = time.time()
    return partid_dict

def calc_mappings():
    start = time.time()
    p = Pool(None) ##uses all available cpu cores, 32 in this case
    p.map(process_single_event, list(range(1000,1100))) #just contents of training-sample for now, i.e first 100 events 
    data_dir = 'mappings'
    all_data = {}
    for folder, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(folder,file)) as file:
                    single_event_info = json.load(file)
                    all_data.update(single_event_info)
    with open('training-sample-mappings.json','w') as outfile:
        json.dump(all_data, outfile)
    end = time.time()
    print('elasped time:'+str(end-start))
    return
    
def get_mappings():
    path = 'training-sample-mappings.json'
    with open(path) as file:
        data = json.load(file)
    return data


# In[18]:


#print(timeit.timeit("calc_mappings()",number=1, setup="from __main__ import calc_mappings"))


# In[2]:


#can prob be further optimzed 
def find_top_hyperstrips(k,toFile=False):
    """
        k: num of hyperstrips we're considering 
        toFile: flag for if we want to write the returned {K:V} to file
        
        
        returns a {(vol_id,layer_id,module_id):(cx,cy,cz)} of superstrips from the first k-most hyperstrips that contain the most tracks   
    """
    
    data = get_mappings()
    hyperstrips_dict = {}
    for key, lists in data.items():
        superstrip_info = []
        hyperstrip_info = [] #hyperstrip is a subset of superstrips for a given track
        for small_dictionaries in lists:
            if 'hit_id' in small_dictionaries:
                del small_dictionaries['hit_id']
            superstrip = (small_dictionaries['volume_id'], small_dictionaries['layer_id'], small_dictionaries['module_id'])
            hyperstrip_info.append(superstrip)

        hyperstrip = tuple(hyperstrip_info)
        #print(hyperstrip)
        if hyperstrip in hyperstrips_dict:
            #print('found one!')
            hyperstrips_dict[hyperstrip] += 1
        else:
            hyperstrips_dict[hyperstrip] = 1

    sorted_hyperstrips = sorted(hyperstrips_dict.items(), key=lambda kv: kv[1], reverse=True)

    for idx, items in enumerate(sorted_hyperstrips):
        hyperstrip, count = items
        if len(hyperstrip) > 1 :
            start = idx
            #print(sorted_hyperstrips[start: start + 500])
            someslice = sorted_hyperstrips[start:start+k]
            break

    #creates a dicitonary for frequency of hits in hyperstrips
    lengths_freq = {}
    for elem in someslice:
        hyperstrip, number_of_tracks = elem 
        hyperstrip_len = len(hyperstrip)
        if hyperstrip_len in lengths_freq:
            lengths_freq[hyperstrip_len] += 1
        else:
            lengths_freq[hyperstrip_len] = 1

    detector_info_path = 'data/detectors.csv'
    detectors = pd.read_csv(detector_info_path)

    superstrip_locs = {}
    json_data = {}
    for hyperstrip in someslice:
        subset_of_strips, tracks = hyperstrip
        for superstrip in subset_of_strips:
            vol, lay, mod = superstrip
            x = detectors.loc[(detectors['volume_id'] == vol) & (detectors['layer_id'] == lay) & (detectors['module_id'] == mod)]['cx'].item()
            y = detectors.loc[(detectors['volume_id'] == vol) & (detectors['layer_id'] == lay) & (detectors['module_id'] == mod)]['cy'].item()
            z = detectors.loc[(detectors['volume_id'] == vol) & (detectors['layer_id'] == lay) & (detectors['module_id'] == mod)]['cz'].item()
            superstrip_locs[superstrip] = (x,y,z) 
            json_data[str(superstrip)] = (x,y,z)
    if toFile:
        with open('top-hyperstrips.json','w') as outfile:
            json.dump(json_data, outfile)
    return superstrip_locs
