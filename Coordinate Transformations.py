
# coding: utf-8

# In[126]:


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
import random


# In[12]:


def load_data_single_event(event_number):
    file_name = 'event00000' + str(event_number)
    event_id = file_name
    hits, cells, particles, truth = load_event('data/train_sample/'+event_id)
    return hits, cells, particles, truth


# In[13]:


hits, cells, particles, truth = load_data_single_event(1000)


# In[14]:


hits.tail()


# In[15]:


for row in hits.itertuples():
    x = row.__getattribute__('x')
    y = row.__getattribute__('y')
    z = row.__getattribute__('z')


# In[142]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

"""
ax = Axes3D( plt.figure(figsize=(20,20)))
sample = hits.sample(10000) ## pandas function, randomly samples given amount
layers = hits.layer_id.unique()
for layer in layers:
    l = sample[sample.layer_id == layer]
    ax.scatter(l.z, l.x, l.y, s=5, label='Layer '+str(layer), alpha=0.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.legend()
# These two added to widen the 3D space
ax.scatter(3000,3000,3000, s=0)
ax.scatter(-3000,-3000,-3000, s=0)
plt.show()
"""

# In[28]:


max_x = hits['x'].max()
max_y = hits['y'].max()
max_z = hits['z'].max()




# In[47]:


def cartesian_to_3d_polar(x,y,z):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    s  = np.sin(phi)
    c  = np.cos(phi)
    return r, phi, z


# In[116]:


ax = Axes3D( plt.figure(figsize=(20,20)))
sample = hits.sample(10000) ## pandas function, randomly samples given amount
"""
for row in sample.itertuples():
    ox = row.__getattribute__('x')
    oy = row.__getattribute__('y')
    oz = row.__getattribute__('z')
    r, phi, z,= cartesian_to_3d_polar(ox,oy,oz)
    ax.scatter(r, phi, z, alpha=0.5, c='0.0')
ax.set_xlabel('r (mm)')
ax.set_ylabel('phi (radians)')
ax.set_zlabel('z (mm)')
ax.legend()
# These two added to widen the 3D space
#ax.scatter(100,2*np.pi,100, s=0)
#ax.scatter(-100,-2*np.pi,-100, s=0)
plt.show()
"""

# In[52]:


#test=sample.head()


# In[53]:


#test


# In[54]:


#test['xyz'] = test[['x','y','z']].values.tolist()


# In[58]:


#grouped = test.groupby(['volume_id','layer_id'])['xyz'].apply(list).to_frame()


# In[89]:


#grouped2 = test.groupby(['volume_id','layer_id'])


# In[92]:


#list(grouped2.groups.keys())


# In[83]:

"""
axes = Axes3D(plt.figure(figsize=(20,20)))
unique
for (idx, row) in grouped.iterrows():
    #print(type(row))
    xyz = np.array(row['xyz'])
    x,y,z = xyz[:,0], xyz[:,1], xyz[:,2]
    axes.plot(x,y,z, linewidth=0.5)
    axes.scatter
#axes.text(x[0], y[0], z[0], str(idx), None)

plt.show()

"""
# In[101]:

"""
unique_groups
"""

# In[143]:


grouped_sample = sample.groupby(['volume_id','layer_id'])
unique_groups = list(grouped_sample.groups.keys())
sample['xyz'] = sample[['x','y','z']].values.tolist()
grouped = sample.groupby(['volume_id','layer_id'])['xyz'].apply(list).to_frame()



color_map = {}
for idx, group in enumerate(unique_groups):
    color_map[group] = (round(random.uniform(0.0,0.99),4),round(random.uniform(0.0,0.99),4),round(random.uniform(0.0,0.99),4))

axes = Axes3D(plt.figure(figsize=(20,20)))
for (idx, row) in grouped.iterrows():
    #print(type(row))
    xyz = np.array(row['xyz'])
    x,y,z = xyz[:,0], xyz[:,1], xyz[:,2]
    r, phi, z = cartesian_to_3d_polar(x,y,z)
    #print(color_map[row.name])
    axes.scatter(r,phi,z, c = color_map[row.name])
#axes.text(x[0], y[0], z[0], str(idx), None)

plt.show()



# In[134]:


color_map
