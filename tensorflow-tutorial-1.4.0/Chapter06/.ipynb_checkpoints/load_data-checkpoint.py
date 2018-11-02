#!/usr/bin/env python
# coding: utf-8

# In[4]:


import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

INPUT_DATA = '../../datasets/flower_processed_data.npy'

def main():
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    
    print len(training_images), len(training_images[0]), len(training_images[0][0])
    
    print training_labels
    
    print validation_labels
    print testing_labels
    

if __name__ == '__main__':
    main()


# In[ ]:




