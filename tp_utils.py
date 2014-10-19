# tp_utils.py
#trans256_name = 'trans256'
#trans256 = __import__(trans256_name)

import sys
sys.path.append('/projects/sciteam/joi/distfiles/Imaging-1.1.7/build/lib.linux-\
x86_64-2.6')
import os
import Image
import cPickle
import tempfile
import numpy as np
# import matplotlib.pyplot as plt
from StringIO import StringIO
from math import floor
from time import time
from progress import progress
import jstore

# Version 2 is better
def ls_r_jpeg2(dir_path,verbose=True,search=None):
    ''' ls_r_jpeg2(dir_path)
    Recursively lists JPEG files.
    I've only tried it with the imagenet training, validation, and test directories.
    '''
    jpeg_list = list()
    if verbose: print 'Crawling directories...'
    for r,d,files in os.walk(dir_path):
        for file in files:
            if file.endswith('.JPEG'):
                if search:
                    if search in file.split('.')[0].split('/')[-1]:
                        jpeg_list.append(os.path.abspath('%s/%s' %(r,file)))
                else:
                    jpeg_list.append(os.path.abspath('%s/%s' %(r,file)))

    jpeg_list = np.sort(jpeg_list)
        
    if verbose: print 'Found %d jpeg files' % len(jpeg_list)
    return jpeg_list

#def test_data(data):
#    image = np.dstack((data[0+224*224*0:50176+224*224*0].reshape(224,224),data[0+224*224*1:50176+224*224*1].reshape(224,224),data[0+224*224*2:50176+224*224*2].reshape(224,224)))
#    plt.imshow(np.uint8(image))
#    plt.show()
    
def batches_from_list(image_list,numBatches,imagesInBatch):
    batches = []
    for i in xrange(0,numBatches):
        start = imagesInBatch*(i)
        end = imagesInBatch*(i+1)
        batches.append(image_list[start:end])
        
    return batches

def crop_n_flip(array,size,offset,flip):
    xoffset = offset[0]
    yoffset = offset[1]
    temp = array[0+xoffset:size+xoffset,0+yoffset:size+yoffset,:]
    if flip:
        vect = np.concatenate((temp[:,::-1,0].flatten(),temp[:,::-1,1].flatten(),temp[:,::-1,2].flatten()))
    elif not flip:
        vect = np.concatenate((temp[:,:,0].flatten(),temp[:,:,1].flatten(),temp[:,:,2].flatten()))
    return vect

def write_class_list(wnid_order_filename, key_filename, class_list_filename):
    wnid_dict = _make_wnid_dict(wnid_order_filename)
    _save_class_list(key_filename, wnid_dict, class_list_filename)
    
def read_class_list(class_list_filename):
    f = open(class_list_filename,'r')

    values = list()
    for line in f:
        values.append(line.split('\n')[0])
    f.close()
    
    return np.array(values,dtype=np.uint16)

def read_class_list_bin(header_filename):
    header_file = open(header_filename,'rb')
    header_string = header_file.read()
    header_file.close()
    header = np.fromstring(header_string,dtype=np.uint64)
    num_jpegs = np.uint32(header.shape[0]/5.0)
    header = header.reshape((num_jpegs,5)).T
    labels = header[1,:]
    return labels

def _make_wnid_dict(wnid_order_filename):
    f = open(wnid_order_filename,'r')
    values = list()
    for line in f:
        values.append(line.split('\n')[0])
    f.close()
    
    wnid_dict = dict()
    for i, value in enumerate(values):
        wnid_dict[value] = i+1
    
    return wnid_dict

def _save_class_list(key_filename, wnid_dict, class_list_filename):
    keys = jstore._unpickle(key_filename)

    class_list = list()
    for key in keys:
        wnid = key.split('_')[0] #WNID (WordNet ID) is the official name
        class_list.append(wnid_dict[wnid])
    
    f = open(class_list_filename,'w')
    for i in class_list:
        f.write('%d\n' % i)
    f.close()

def initialize_jstore_mem(data_dir,test,type_='not-bin'):
    class_dict = {0:'train',1:'val',2:'test'}
    main_filename = '%s/ilsvrc2012_%s' %(data_dir,class_dict[test])
    class_filename = '%s_labels.txt' % main_filename

    print 'Looking for jstore: %s' % main_filename
    dataStore = jstore.MemJstore(main_filename,type_=type_)
    if type_ == 'not-bin': # depricated
        labelStore = read_class_list(class_filename)
    elif type_ == 'bin':
        header_filename = '%s_header.bin' % main_filename
        labelStore = read_class_list_bin(header_filename)
        smallest = np.min(labelStore)
        if smallest == 0:
            labelStore+=1
            print 'smallest label set to 1'
        elif smallest == 1:
            print 'smallest label is 1'
        else:
            print 'something is very wrong!!!'

    return dataStore, labelStore

def initialize_jstore(data_dir,test,type_='not-bin'):
    class_dict = {0:'train',1:'val'}
    main_filename = '%s/ilsvrc2012_%s' %(data_dir,class_dict[test])
    class_filename = '%s_labels.txt' % main_filename

    print 'Looking for jstore: %s' % main_filename
    dataStore = jstore.Jstore(main_filename,type_=type_)
    if type_=='not-bin': # depricated
        labelStore = read_class_list(class_filename)
    elif type_=='bin':
        header_filename = '%s_header.bin' % main_filename
        labelStore = read_class_list_bin(header_filename)
        smallest = np.min(labelStore)
        if smallest == 0:
            labelStore+=1
            print 'smallest label set to 1'
        elif smallest == 1:
            print 'smallest label is 1'
        else:
            print 'something is very wrong!!!'

    return dataStore, labelStore

def make_batch_jstore(dataStore, labelStore, batch, average):
    imagesInBatch = len(batch)
    data = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    labels = np.zeros((1,imagesInBatch),dtype=np.single)

    for count, i in enumerate(batch):
        xoffset = np.random.randint(0,32)
        yoffset = np.random.randint(0,32)
        flip = np.random.randint(0,2)
        average_vect = crop_n_flip(average,224,[xoffset,yoffset],flip)
        image_vect = crop_n_flip(dataStore.get(i),224,[xoffset,yoffset],flip)
        data[count,:] = image_vect - average_vect
        labels[0,count] = labelStore[i]
    
    return np.transpose(data), labels

def make_batch_jstore_better(dataStore, labelStore, batch, average, chan1, chan2, chan3):
    imagesInBatch = len(batch)
    data = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    labels = np.zeros((1,imagesInBatch),dtype=np.single)

    crop = crop_maker() # has all the logic for making, and testing crops

    for count, i in enumerate(batch):
        A, b = crop.get()
        rgb = np.random.normal(scale=0.3,size=(3,1)).astype(np.float32)
        alpha = 0.0
        mode = 0 # 0: channels, 1: tint
        img_dict = {'src':dataStore.get(i), 'dst':data[count,:], 'A':A, 'b':b.T, 'rgb':rgb, 'alpha':alpha,
        'mean':average, 'mode':mode, 'chan1':chan1, 'chan2':chan2, 'chan3':chan3 }
        img_list = [img_dict]
        trans256.transform(img_list)
        labels[0,count] = labelStore[i]

    return np.transpose(data), labels

def make_multiview_batch_jstore(dataStore, labelStore, batch, average):
    # given dataStore, and list of keys to jpegs
    # converts jpegs into numpy array of singles
    # stores those in the columns of a matrix: data
    # returns data
    print 'make batch and labels: multiview'
    imagesInBatch = len(batch)

    tic = time()

    data1 = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    data2 = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    data3 = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    data4 = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    data5 = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    data6 = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    data7 = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    data8 = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    data9 = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    data10 = np.zeros((imagesInBatch,224*224*3),dtype=np.single)
    labels = np.zeros((1,imagesInBatch),dtype=np.single)
    count = 0

    for count, i in enumerate(batch):
        image_array = dataStore.get(i)        
        image_vect1 = crop_n_flip(image_array,224,[0,0],0)
        image_vect2 = crop_n_flip(image_array,224,[16,16],0)
        image_vect3 = crop_n_flip(image_array,224,[0,32],0)
        image_vect4 = crop_n_flip(image_array,224,[32,0],0)
        image_vect5 = crop_n_flip(image_array,224,[32,32],0)
        image_vect6 = crop_n_flip(image_array,224,[0,0],1)
        image_vect7 = crop_n_flip(image_array,224,[16,16],1)
        image_vect8 = crop_n_flip(image_array,224,[0,32],1)
        image_vect9 = crop_n_flip(image_array,224,[32,0],1)
        image_vect10 = crop_n_flip(image_array,224,[32,32],1)
        
        average_vect = crop_n_flip(average,224,[16,16],0)
        
        data1[count,:] = image_vect1 - average_vect
        data2[count,:] = image_vect2 - average_vect
        data3[count,:] = image_vect3 - average_vect
        data4[count,:] = image_vect4 - average_vect
        data5[count,:] = image_vect5 - average_vect
        data6[count,:] = image_vect6 - average_vect
        data7[count,:] = image_vect7 - average_vect
        data8[count,:] = image_vect8 - average_vect
        data9[count,:] = image_vect9 - average_vect
        data10[count,:] = image_vect10 - average_vect
        
        labels[0,count] = labelStore[i]
        
        count = count + 1
    data = np.concatenate((data1,data2,data3,data4,data5,data6,data7,data8,data9,data10),axis=0)
    labels = np.concatenate((labels,labels,labels,labels,labels,labels,labels,labels,labels,labels),axis=1)
    toc = time()
    
    print 'making the batch took: %f sec.' % (toc-tic)
    
    return np.transpose(data), labels

class crop_maker():
    def __init__(self):
        # bounding box
        self.box_coords = np.array([[0,0,255,255,0],[0,255,255,0,0]])
        self.box_offset = np.ones((2,1))*128
        
        # initial crop box
        self.crop_coords = np.array([[0,0,224,224,0],[0,224,224,0,0]])
        self.crop_offset = np.ones((2,1))*112
    
    def get(self):
        success = 0
        while not success:
            # generate transformation values
            shift = np.random.randint(low=-16,high=16,size=(2,1))
            deg = np.random.randint(low=-10,high=10)
            scale = np.random.uniform(low=10/11.0,high=11/10.0)
            flip = np.random.randint(2)
            
            # generate transformation matrix using generated values
            A = self._make_mat(deg,scale,flip)
            
            # calculate shift including offsets
            b = -np.dot(A,self.crop_offset)+self.box_offset+shift
            b = b.astype(np.float32)
            
            # apply coordinate transformation
            new_coords = np.dot(A,self.crop_coords) + b
            
            # test coordinate transformation
            success = self._test_crop(new_coords)
        return A, b
    
    def _make_mat(self,theta,scale,flip):
        theta = theta*(np.pi/180)
        R = np.zeros((2,2),dtype=np.float32)
        S = np.zeros((2,2),dtype=np.float32)
        if flip:
            S[0,0] = -scale
        else:
            S[0,0] = scale
        S[1,1] = scale
        R[0,0] = np.cos(theta)
        R[0,1] = -np.sin(theta)
        R[1,0] = np.sin(theta)
        R[1,1] = np.cos(theta)
        T = np.dot(S,R)
        return T
    
    def _test_crop(self,crop_coords):
        boxx  = self.box_coords[0,:]
        boxy  = self.box_coords[1,:]
        cropx = crop_coords[0,:]
        cropy = crop_coords[1,:]
        if np.any(cropx<1) or np.any(cropx>254) or np.any(cropy<1) or np.any(cropy>254):
            output = 0
        else:
            output = 1
        return output

# Old stuff. Only use in case of emergency.
def ls_r_jpeg(dir_name):
    # given a directory name full of sub-directories
    # returns of list of jpegs in all sub-directories
    
    image_list = []

    sub_dir_list = os.listdir(dir_name)

    for iSub_dir in sub_dir_list:
        if not iSub_dir.endswith('.ipynb'):
            sub_dir_path = '%s/%s' % (dir_name,iSub_dir)
            file_list = os.listdir(sub_dir_path)
            for iFile in file_list:
                if iFile.endswith('.JPEG'):
                    image_path = '%s/%s' % (sub_dir_path,iFile)
                    image_list.append(image_path)
                
    return image_list

def load_images(image_list,test):
    # given a list of jpegs
    # loads them into memory using hdf5 as: dataStore
    # returns dataStore, and list of keys to access the jpegs
    file_name = 'dataStore%d.hdf5' % test
    print '#### %s' % file_name
    dataStore = h5py.File(file_name,driver='core',backing_store=False)
    tic = time()
    dataStore_keys = []

    for iImage in image_list:
        key = iImage.split('.')[0].split('/')[-1] # gross code to get image name
        temp = open(iImage,'rb')
        temp_binary = temp.read()
        dataStore.create_dataset(key,data=temp_binary)
        dataStore_keys.append(key)
    toc = time()
    
    # display time
    print 'loading images into memory took: %f sec.' % (toc-tic)
    print 'success!!!'
    return dataStore, dataStore_keys

def load_images_to_hd(image_list,test):
    # given a list of jpegs
    # loads them into memory using hdf5 as: dataStore
    # returns dataStore, and list of keys to access the jpegs
    #file_name = 'dataStore%d.hdf5' % test
    file_name = '/mnt/ssd/paine/hd_dataStore%d.hdf5' % test
    print '#### %s' % file_name
    dataStore = h5py.File(file_name)
    tic = time()
    dataStore_keys = []

    for iImage in image_list:
        key = iImage.split('.')[0].split('/')[-1] # gross code to get image name
        temp = open(iImage,'rb')
        temp_binary = temp.read()
        dataStore.create_dataset(key,data=temp_binary)
        dataStore_keys.append(key)
    toc = time()
    
    # display time
    print 'loading images into memory took: %f sec.' % (toc-tic)
    print 'success!!!'
    return dataStore, dataStore_keys

def make_batch(dataStore,key_list):
    # given dataStore, and list of keys to jpegs
    # converts jpegs into numpy array of singles
    # stores those in the columns of a matrix: data
    # returns data
    
    tic = time()
    average = np.load(open('/mnt/ssd/paine/average','r'))
    numKeys = len(key_list) # keys correspond to images
    data = np.zeros((numKeys,224*224*3),dtype=np.single)
    count = 0
    for iKey in key_list:
        xoffset = np.random.randint(0,32)
        yoffset = np.random.randint(0,32)
        flip = np.random.randint(0,2)
        temp = np.array(Image.open(StringIO(dataStore[iKey].value)).convert('RGB'),dtype=np.single)
        temp2 = temp[0+xoffset:224+xoffset,0+yoffset:224+yoffset,:]
        average2 = temp[0+xoffset:224+xoffset,0+yoffset:224+yoffset,:]
        average_vect = np.concatenate((average2[:,:,0].flatten(),average2[:,:,1].flatten(),average2[:,:,2].flatten()))
        if flip:
            vect = np.concatenate((temp2[:,::-1,0].flatten(),temp2[:,::-1,1].flatten(),temp2[:,::-1,2].flatten())) - average_vect
            #vect = np.concatenate((temp[0+xoffset:224+xoffset,0+yoffset:224+yoffset,0].flatten(),temp[0+xoffset:224+xoffset,0+yoffset:224+yoffset,1].flatten(),temp[0+xoffset:224+xoffset,0+yoffset:224+yoffset,2].flatten()))
        elif not flip:
            vect = np.concatenate((temp2[:,:,0].flatten(),temp2[:,:,1].flatten(),temp2[:,:,2].flatten())) - average_vect
            #vect = np.concatenate((temp[244+xoffset:0+xoffset:-1,224+yoffset:0+yoffset:-1,0].flatten(),temp[224+xoffset:0+xoffset:-1,224+yoffset:0+yoffset:-1,1].flatten(),temp[224+xoffset:0+xoffset:-1,224+yoffset:0+yoffset:-1,2].flatten()))
        #temp = np.array(Image.open(StringIO(dataStore[iKey].value)).convert('RGB'),order='C')
        #temp2 = np.fliplr(np.rot90(temp[0:224,0:224,:],k=3))
        #vect = temp2.T.flatten('C')
        data[count,:] = vect
        count = count + 1
    toc = time()
    
    # print 'making the batch took: %f sec.' % (toc-tic)
    
    return np.transpose(data)

def get_class_dict(dataStore_keys):

    classes = set()
    for iKey in dataStore_keys:
        #class_name = iKey.split('_')[0]
        class_name = iKey.split('.JPEG')[0].split('/')[-2]
        classes.add(class_name)

    class_dict = {}
    count = 0
    for iClass in classes:
        class_dict[iClass] = count
        count = count + 1
    
    return class_dict

def make_batch_labels(class_dict,batch_keys):

    labels = np.zeros((1,len(batch_keys)),dtype=np.single)

    count = 0
    for iKey in batch_keys:
        class_name = iKey.split('_')[0]
        labels[0,count] = class_dict[class_name]
        count = count + 1
    return labels

def split_image_list(image_list,imagesInBatch,startBatch,endBatch):
    start = (startBatch-1)*imagesInBatch
    end = endBatch*imagesInBatch
    return image_list[start:end]

def initialization(data_dir,seed,imagesInBatch,startBatch,endBatch):
    image_list = ls_r_jpeg(data_dir)
    np.random.seed(seed)
    np.random.shuffle(image_list)
    
    class_dict = get_class_dict(image_list)
    
    images_to_use = split_image_list(image_list,imagesInBatch,startBatch,endBatch)
    
    print len(images_to_use)
    # print images_to_use[0]
    # print images_to_use[-1]
    
    return images_to_use, class_dict

def make_batch_n_labels(dataStore,key_list,class_dict):
    # given dataStore, and list of keys to jpegs
    # converts jpegs into numpy array of singles
    # stores those in the columns of a matrix: data
    # returns data

    tic = time()
    average = np.load(open('/home/compute/paine/average','r'))
    #average = np.load(open('/mnt/ssd/paine/average','r'))
    numKeys = len(key_list) # keys correspond to images
    data = np.zeros((numKeys,224*224*3),dtype=np.single)
    labels = np.zeros((1,len(key_list)),dtype=np.single)
    count = 0

    for iKey in key_list:
        xoffset = np.random.randint(0,32)
        yoffset = np.random.randint(0,32)
        flip = np.random.randint(0,2)
        class_name = iKey.split('_')[0]
        image_array = jpeg_to_array(dataStore[iKey])
        # print class_name
        image_vect = crop_n_flip(image_array,224,[xoffset,yoffset],flip)
        average_vect = crop_n_flip(average,224,[xoffset,yoffset],flip)
        data[count,:] = image_vect - average_vect
        labels[0,count] = class_dict[class_name]
        
        count = count + 1
    toc = time()
    
    # print 'making the batch took: %f sec.' % (toc-tic)
    
    return np.transpose(data), labels

def make_multiview_batch_n_labels(dataStore,key_list,class_dict):
    # given dataStore, and list of keys to jpegs
    # converts jpegs into numpy array of singles
    # stores those in the columns of a matrix: data
    # returns data
    
    print 'make batch and labels: multiview'

    tic = time()
    average = np.load(open('/home/compute/paine/average','r'))
    #average = np.load(open('/mnt/ssd/paine/average','r'))
    numKeys = len(key_list) # keys correspond to images
    data1 = np.zeros((numKeys,224*224*3),dtype=np.single)
    data2 = np.zeros((numKeys,224*224*3),dtype=np.single)
    data3 = np.zeros((numKeys,224*224*3),dtype=np.single)
    data4 = np.zeros((numKeys,224*224*3),dtype=np.single)
    data5 = np.zeros((numKeys,224*224*3),dtype=np.single)
    data6 = np.zeros((numKeys,224*224*3),dtype=np.single)
    data7 = np.zeros((numKeys,224*224*3),dtype=np.single)
    data8 = np.zeros((numKeys,224*224*3),dtype=np.single)
    data9 = np.zeros((numKeys,224*224*3),dtype=np.single)
    data10 = np.zeros((numKeys,224*224*3),dtype=np.single)
    labels = np.zeros((1,len(key_list)),dtype=np.single)
    count = 0

    for iKey in key_list:
        class_name = iKey.split('_')[0]
        image_array = jpeg_to_array(dataStore[iKey])
        
        image_vect1 = crop_n_flip(image_array,224,[0,0],0)
        image_vect2 = crop_n_flip(image_array,224,[16,16],0)
        image_vect3 = crop_n_flip(image_array,224,[0,32],0)
        image_vect4 = crop_n_flip(image_array,224,[32,0],0)
        image_vect5 = crop_n_flip(image_array,224,[32,32],0)
        image_vect6 = crop_n_flip(image_array,224,[0,0],1)
        image_vect7 = crop_n_flip(image_array,224,[16,16],1)
        image_vect8 = crop_n_flip(image_array,224,[0,32],1)
        image_vect9 = crop_n_flip(image_array,224,[32,0],1)
        image_vect10 = crop_n_flip(image_array,224,[32,32],1)
        
        average_vect = crop_n_flip(average,224,[16,16],0)
        
        data1[count,:] = image_vect1 - average_vect
        data2[count,:] = image_vect2 - average_vect
        data3[count,:] = image_vect3 - average_vect
        data4[count,:] = image_vect4 - average_vect
        data5[count,:] = image_vect5 - average_vect
        data6[count,:] = image_vect6 - average_vect
        data7[count,:] = image_vect7 - average_vect
        data8[count,:] = image_vect8 - average_vect
        data9[count,:] = image_vect9 - average_vect
        data10[count,:] = image_vect10 - average_vect
        
        labels[0,count] = class_dict[class_name]
        
        count = count + 1
    data = np.concatenate((data1,data2,data3,data4,data5,data6,data7,data8,data9,data10),axis=0)
    labels = np.concatenate((labels,labels,labels,labels,labels,labels,labels,labels,labels,labels),axis=1)
    toc = time()
    
    print 'making the batch took: %f sec.' % (toc-tic)
    
    return np.transpose(data), labels

def hd_initialization(test):
    file_name = '/mnt/ssd/paine/hd_dataStore%d.hdf5' % test
    print '#### %s' % file_name
    dataStore = h5py.File(file_name,'r')
    key_file_name = '/mnt/ssd/paine/hd_dataStore_keys%d' % test
    dataStore_keys = cPickle.load(open(key_file_name,'r'))
    class_file_name = '/mnt/ssd/paine/hd_class_dict'
    class_dict = cPickle.load(open(class_file_name,'r'))
    
    return dataStore, dataStore_keys, class_dict

def h5py_memory_initialization(test):
    file_name = '/home/compute/paine/hd_dataStore%d.hdf5' % test
    key_file_name = '/home/compute/paine/hd_dataStore_keys%d' % test
    print '#### %s' % file_name
    dataStore, dataStore_keys = h5py_load_to_memory(file_name,key_file_name,test)
    class_file_name = '/home/compute/paine/hd_class_dict'
    class_dict = cPickle.load(open(class_file_name,'r'))
    
    return dataStore, dataStore_keys, class_dict

def h5py_load_to_memory(data_file_name,key_file_name,test):
    dataStore = h5py.File(data_file_name,mode='r')
    dataStore_keys = cPickle.load(open(key_file_name,'r'))
    file_name = 'temp%d' % test
    dataStoreNew = h5py.File(file_name,driver='core',backing_store=False)
    
    print 'Loading images to memory...'
    
    perc = floor(len(dataStore_keys)/100)
    ptic = time()
    tic = time()
    pcount = 0
    count = 0
    
    for iKey in dataStore_keys:
        if count == perc:
            toc = time()
            print '%.2f%%, %.2f' % (pcount/perc,toc-tic)
            tic = time()
            count = 0
        dataStoreNew[iKey] = dataStore[iKey].value
        pcount = pcount+1
        count = count+1
    ptoc = time()
    
    print 'Loading took %.2f' % (ptoc-ptic)
    
    dataStore.close()
    return dataStoreNew, dataStore_keys

def h5py_load_to_memory2(data_file_name,key_file_name,test):
    file = open(data_file_name,'rb')
    tic = time()
    print 'Loading images to memory...'
    data = file.read()
    file.close()
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(data)
    temp.close()
    dataStoreNew = h5py.File(temp.name,driver='core',backing_store=False)
    dataStore_keys = cPickle.load(open(key_file_name,'r'))
    toc = time()

    print 'Loading took %.2f' % (toc-tic)

    return dataStoreNew, dataStore_keys

def jpeg_to_array(dataset):
    return np.array(Image.open(StringIO(dataset.value)).convert('RGB'))
