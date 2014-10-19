# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from data import *
import numpy.random as nr
import numpy as np
import random as r
import tp_utils
import cPickle
import Image
import copy
import tp_image_tools as it
import tp_detect_tools as dt
import tp_detect_tools_alt as dt_alt
import tp_tools as tp
#import matplotlib.pyplot as plt
from time import time

class JstoreDetectAltDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self,data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        self.img_size = 224
        self.num_views = 5*2
        self.imagesInBatch = 128
        self.test = test

        self.multiview = dp_params['multiview_test'] and self.test

    def tp_init(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):

        # set this part up to use image_tools and classify tools
        class_dict = {0:'train',1:'test'}
        filename = os.path.join(data_dir,class_dict[self.test])
        self.jStore = it.Jstore(filename)
        self.labelMaskStore = dt_alt.LocationMaskStore(filename)
        average_filename = '%s/average.npy' % data_dir

        self.average_vect = it.load_average(average_filename)

        #class_id = 0 # temp value to test on single machine
        #self.hits, self.misses = self.labelMaskStore.search(class_id)  

        np.random.seed(42)
        self.image_list = range(self.jStore.num_files)
        np.random.shuffle(self.image_list)
        self.batches = tp.batches_from_list(self.image_list,self.imagesInBatch)

    def make_batch(self,batch):
        data = np.zeros((self.imagesInBatch,224*224*3),dtype=np.single)
        labels = np.zeros((self.imagesInBatch,29*29*16),dtype=np.single)

        for count, im_id in enumerate(batch):
            mask, crop = self.labelMaskStore.get(im_id)
            image = self.jStore.get(im_id,crop)
            data[count,:] = it.vectorize(image) - self.average_vect
            labels[count,:] = dt_alt.vectorize(mask)

        return np.transpose(data), np.transpose(labels)

    def get_next_batch(self):
        self.advance_batch()

        epoch = self.curr_epoch
        batch_num = self.batch_idx      

        if self.multiview:
            print 'ERROR!!!'
        else:
            data,labels = self.make_batch(self.batches[self.batch_idx])
        
        data = np.require(data,requirements='C')

        labels = np.require(labels,requirements='C')

        return epoch, batch_num, [data, labels]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        #print 'Data_dims: %s' % idx
        return self.img_size**2 * self.num_colors if idx == 0 else 29*29*16

    def get_num_classes(self):
        #print len(self.tp_class_dict)
        return 29*29*16

    def advance_batch(self):
        #print 'Im advancing the batch.'
        self.batch_idx = self.get_next_batch_idx()

        if self.batch_idx == 0: # we wrapped
            #print 'Im advancing the epoch!!!'
            np.random.shuffle(self.image_list)
            self.batches = tp.batches_from_list(self.image_list,self.imagesInBatch)

            self.curr_epoch += 1

class JstoreDetectDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self,data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        self.img_size = 224
        self.num_views = 5*2
        self.imagesInBatch = 128
        self.test = test

        self.multiview = dp_params['multiview_test'] and self.test

    def tp_init(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        # set this part up to use image_tools and classify tools
        class_dict = {0:'train',1:'test'}
        filename = os.path.join(data_dir,class_dict[self.test])
        self.jStore = it.Jstore(filename)
        self.labelMaskStore = dt.LocationMaskStore(filename)
        average_filename = '%s/average.npy' % data_dir

        self.average_vect = it.load_average(average_filename)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.class_id = rank-1

        print "Class: %s" % self.labelMaskStore.object_names[self.class_id]
        self.hits, self.misses = self.labelMaskStore.search(self.class_id)  

        np.random.seed(42)

        np.random.shuffle(self.hits)
        np.random.shuffle(self.misses)
        self.image_list = np.concatenate((self.hits,self.misses[0:len(self.hits)]))
        np.random.shuffle(self.image_list)
        self.batches = tp.batches_from_list(self.image_list,self.imagesInBatch)
        self.batch_range = range(0,len(self.batches))

    def make_batch(self,batch):
        data = np.zeros((self.imagesInBatch,224*224*3),dtype=np.single)
        labels = np.zeros((self.imagesInBatch,24*24),dtype=np.single)

        for count, im_id in enumerate(batch):
            mask, crop = self.labelMaskStore.get(im_id,class_id=self.class_id) # temp value to test on single machine
        #print crop['x']
            #print crop['y']
            #print crop['width_x']
            #print crop['width_y']
            #print im_id
            image = self.jStore.get(im_id,crop)
            #plt.subplot(1,2,1)
            #plt.imshow(image)
            #plt.subplot(1,2,2)
            #plt.imshow(mask,vmin=0,vmax=1)
            #plt.show()
            data[count,:] = it.vectorize(image) - self.average_vect
            labels[count,:] = mask.flatten()

        return np.transpose(data), np.transpose(labels)

    def get_next_batch(self):
        self.advance_batch()

        epoch = self.curr_epoch
        batch_num = self.batch_idx      

        if self.multiview:
            print 'ERROR!!!'
        else:
            data,labels = self.make_batch(self.batches[self.batch_idx])
        
        data = np.require(data,requirements='C')

        labels = np.require(labels,requirements='C')

        return epoch, batch_num, [data, labels]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        #print 'Data_dims: %s' % idx
        return self.img_size**2 * self.num_colors if idx == 0 else 24*24

    def get_num_classes(self):
        #print len(self.tp_class_dict)
        return 24*24

    def advance_batch(self):
        #print 'Im advancing the batch.'
        self.batch_idx = self.get_next_batch_idx()

        if self.batch_idx == 0: # we wrapped
            #print 'Im advancing the epoch!!!'
            np.random.shuffle(self.hits)
            np.random.shuffle(self.misses)
            self.image_list = np.concatenate((self.hits,self.misses[0:len(self.hits)]))
            np.random.shuffle(self.image_list)
            self.batches = tp.batches_from_list(self.image_list,self.imagesInBatch)

            self.curr_epoch += 1

class ImageNetJstoreDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self,data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        self.img_size = 224
        self.num_views = 5*2
        imagesInBatch = 128
        numBatches = len(batch_range)

        self.multiview = dp_params['multiview_test'] and test

    def tp_init(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        print '#### Help'
        imagesInBatch = 128
        numBatches = len(batch_range)

        tp_dataStore, tp_labelStore = self.initialize(data_dir,test)
        average_filename = '%s/average.npy' % data_dir
        #chan1_filename   = '%s/chan1m.npy' % data_dir
        #chan2_filename   = '%s/chan2m.npy' % data_dir
        #chan3_filename   = '%s/chan3m.npy' % data_dir
        tp_average = np.load(average_filename).astype(np.float32)
        print tp_average.dtype        
        #tp_chan1   = np.load(chan1_filename).astype(np.float32)
        #tp_chan2   = np.load(chan2_filename).astype(np.float32)
        #tp_chan3   = np.load(chan3_filename).astype(np.float32)

        tp_image_list = np.array(range(tp_dataStore.num_jpegs))
        np.random.shuffle(tp_image_list)
        tp_batches = tp_utils.batches_from_list(tp_image_list,numBatches,imagesInBatch)

        tp_batch_dic = {}
        count = 0
        for iBatch in batch_range:
            tp_batch_dic[iBatch] = count
            count = count + 1

        self.tp_imagesInBatch = imagesInBatch
        self.tp_numBatches = numBatches
        self.tp_dataStore = tp_dataStore
        self.tp_labelStore = tp_labelStore # because it goes from 0 to (n-1) instead of 1 to np.
        print np.min(tp_labelStore)
        print np.max(tp_labelStore)
        self.tp_image_list = tp_image_list
        self.tp_batches = tp_batches
        self.tp_batch_dic = tp_batch_dic
        self.tp_average = tp_average
        #self.tp_chan1 = tp_chan1
        #self.tp_chan2 = tp_chan2
        #self.tp_chan3 = tp_chan3

    def initialize(self, data_dir, test):
        dataStore, labelStore = tp_utils.initialize_jstore(data_dir,test)
        return dataStore, labelStore

    def make_batch(self,batch_ind):
        #data,labels = tp_utils.make_batch_jstore_better(self.tp_dataStore, self.tp_labelStore, self.tp_batches[batch_ind], self.tp_average, self.tp_chan1, self.tp_chan2, self.tp_chan3)
        data,labels = tp_utils.make_batch_jstore(self.tp_dataStore, self.tp_labelStore, self.tp_batches[batch_ind], self.tp_average)
        return data, labels

    def get_next_batch(self):
        self.advance_batch()

        epoch = self.curr_epoch        
        batch_num = self.curr_batchnum

        tp_batch_ind = self.tp_batch_dic[batch_num] 
    
        # print 'tp-info, batch_name: %d' % tp_batch_ind
        if self.multiview:
            data, labels = tp_utils.make_multiview_batch_jstore(self.tp_dataStore, self.tp_labelStore, self.tp_batches[tp_batch_ind], self.tp_average)
        else:
            data,labels = self.make_batch(tp_batch_ind-1)
        
        data = np.require(data,requirements='C')

        labels = np.require(labels,requirements='C')

        return epoch, batch_num, [data, labels]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        #print 'Data_dims: %s' % idx
        return self.img_size**2 * self.num_colors if idx == 0 else 1

    def get_num_classes(self):
        #print len(self.tp_class_dict)
        return 1000
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return np.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=np.single)

    def advance_batch(self):
        #print 'Im advancing the batch.'
        self.batch_idx = self.get_next_batch_idx()
        self.curr_batchnum = self.batch_range[self.batch_idx]

        if self.batch_idx == 0: # we wrapped
            #print 'Im advancing the epoch!!!'
            np.random.shuffle(self.tp_image_list)
            self.tp_batches = tp_utils.batches_from_list(self.tp_image_list,self.tp_numBatches,self.tp_imagesInBatch)

            self.curr_epoch += 1

class ImageNetMemJstoreDataProvider(ImageNetJstoreDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        ImageNetJstoreDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

    def initialize(self, data_dir, test):
        dataStore, labelStore = tp_utils.initialize_jstore_mem(data_dir,test)
        return dataStore, labelStore

### Old Stuff. Only use in case of emergency.

class ImageNetDataH5pyMemoryProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        self.img_size = 224
        self.num_views = 5*2

        self.multiview = dp_params['multiview_test'] and test


    def tp_init(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        imagesInBatch = 128
        numBatches = len(batch_range)
        tp_dataStore, tp_dataStore_keys, tp_class_dict = tp_utils.h5py_memory_initialization(test)

        #tp_images_to_use, tp_class_dict = tp_utils.initialization(data_dir,42,imagesInBatch,batch_range[0],batch_range[-1])
        #tp_dataStore, tp_dataStore_keys = tp_utils.load_images_to_hd(tp_images_to_use,test)
        
        tp_batches = tp_utils.batches_from_list(tp_dataStore_keys,numBatches,imagesInBatch)

        tp_batch_dic = {}
        count = 0
        for iBatch in batch_range:
            tp_batch_dic[iBatch] = count
            count = count + 1

        self.tp_imagesInBatch = imagesInBatch
        self.tp_numBatches = numBatches
        self.tp_class_dict = tp_class_dict
        self.tp_dataStore = tp_dataStore
        self.tp_dataStore_keys = tp_dataStore_keys
        self.tp_batches = tp_batches
        self.tp_batch_dic = tp_batch_dic    

    def get_next_batch(self):
        self.advance_batch()

        epoch = self.curr_epoch        
        batch_num = self.curr_batchnum

        tp_batch_ind = self.tp_batch_dic[batch_num]

        # print 'tp-info, batch_name: %d' % tp_batch_ind
        if self.multiview:
            data,labels = tp_utils.make_multiview_batch_n_labels(self.tp_dataStore,self.tp_batches[tp_batch_ind],self.tp_class_dict)
        else:
            data,labels = tp_utils.make_batch_n_labels(self.tp_dataStore,self.tp_batches[tp_batch_ind],self.tp_class_dict)

        data.shape

        # data = tp_utils.make_batch(self.tp_dataStore,self.tp_batches[tp_batch_ind])
        # labels = tp_utils.make_batch_labels(self.tp_class_dict,self.tp_batches[tp_batch_ind])
        #dic = {'data':data,'labels':labels}
        
        data = np.require(data,requirements='C')

        #tp_utils.test_data(data[:,0])

        labels = np.require(labels,requirements='C')

        return epoch, batch_num, [data, labels]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        #print 'Data_dims: %s' % idx
        return self.img_size**2 * self.num_colors if idx == 0 else 1

    def get_num_classes(self):
        #print len(self.tp_class_dict)
        return 1000
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return np.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=np.single)

    def advance_batch(self):
        #print 'Im advancing the batch.'
        self.batch_idx = self.get_next_batch_idx()
        self.curr_batchnum = self.batch_range[self.batch_idx]

        if self.batch_idx == 0: # we wrapped
            #print 'Im advancing the epoch!!!'
            np.random.shuffle(self.tp_dataStore_keys)
            # print self.tp_dataStore_keys[0]
            batches = tp_utils.batches_from_list(self.tp_dataStore_keys,self.tp_numBatches,self.tp_imagesInBatch)
            self.tp_batches = batches
            #print '\n'
            #print self.tp_batches[0]
            self.curr_epoch += 1
