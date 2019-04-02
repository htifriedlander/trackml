
# coding: utf-8

# In[91]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import random
from trackml.dataset import load_event, load_dataset
#get_ipython().run_line_magic('run', 'utils.ipynb')
#get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


np.load ('./seeds/training_event_1050.npy')


# In[3]:


seeds = np.load ('./seeds/training_event_1050.npy')


# In[4]:


def load_single_train_event(event_number):
    train_dir = 'data/train_sample/'
    hits, cells, particles, truth = load_event(train_dir+'event00000'+str(event_number))
    return hits, cells, particles, truth

hits, cells, particles, truth = load_single_train_event(1050)


# In[5]:


#array_of_tracks = rpzlv_from_seeds(seeds, hits)


# In[6]:


#np.random.shuffle(array_of_tracks)


# In[7]:


#ax = Axes3D(plt.figure(figsize=(10,10)))
#for track in array_of_tracks[:150]:
#    r, phi, z = track[:,0], track[:,1], track[:,2]
#    ax.scatter(r, phi, z)
#    ax.plot(r, phi, z)
#ax.view_init(0,90)
#ax.set_xlabel('r (mm)')
#ax.set_ylabel('phi (radians)')
#ax.set_zlabel('z (mm)')
#plt.show()


# In[39]:


hft = np.load ('./hits_from_tracks.npy')


# In[40]:


hft.shape


# In[74]:

abs(phi) > np.pi/2

shift_idxs = np.argwhere(abs(hft[:,:,1][:,0]) > np.pi/2).flatten()


#input is of shape batch_size x 10
def quadrant_shift_vectorized (phis):

    np.where(phis[:,0]<0,2*np.pi+)
    if (phi < 0):
        return 2*np.pi + phi
    else:
        return phi

phi_diffs = np.diff(hft[:,:,1])
phi_diff_tot = np.absolute(np.sum(phi_diffs,axis=1))


# In[75]:


#phi_info = pd.DataFrame(data=phi_diff_tot)


# In[76]:


#phi_info.describe()


# In[86]:


bad_idxs = np.argwhere(phi_diff_tot > 3)


# In[87]:


bad_tracks = hft[bad_idxs.flatten()]


# In[88]:


bad_tracks.shape


# In[89]:


np.random.shuffle(hft)


# In[92]:


ax = Axes3D(plt.figure(figsize=(10,10)))
for track in bad_tracks:
    r, phi, z = track[:,0], track[:,1], track[:,2]
    ax.scatter(r, phi, z)
    ax.plot(r, phi, z)
ax.view_init(0,0)
ax.set_xlabel('r (mm)')
ax.set_ylabel('phi (radians)')
ax.set_zlabel('z (mm)')
plt.show()
