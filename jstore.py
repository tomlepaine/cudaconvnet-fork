# jstore.py
# The Jstore object makes loading and reading images from a jstore file easy.
# The MemJstore object extends this to in memory operation
# The SPEED_TEST function reports the average image-read time
# The MAKE_JSTORE function saves the jpegs to a jstore file

import sys
sys.path.append('/projects/sciteam/joi/distfiles/Imaging-1.1.7/build/lib.linux-\
x86_64-2.6')
import tp_utils
from StringIO import StringIO
import Image
from time import time
import cPickle
import numpy as np

# Ideal interface data['key']
# Current interface data.get(index)

class MemJstore():
    def __init__(self,main_filename,type_='not-bin'):
        self.main_filename = main_filename
        print 'Loading jstore file...'
        if type_=='not-bin':
            # self._mem_file, self.keys, self._offset_array = _mem_load_jstore(main_filename)
            self._mem_file, self._offset_array = _mem_load_jstore(main_filename)
        elif type_=='bin':
            self._mem_file, self._offset_array = _mem_load_jstore_bin(main_filename)
        self._size_array = np.uint32(np.diff(self._offset_array))
        self.num_jpegs = len(self._size_array)
    
    def get(self,i):
        ''' MemJstore.get(i)
        Finds the i-th JPEG in the JSTORE, and returns it as a numpy array.
        '''
        return _read_string(self._mem_file,self._offset_array[i],self._size_array[i])
    
    def get_rand(self):
        ''' MemJstore.get_rand()
        Finds a random JPEG in the JSTORE, and returns it as a numpy array.
        '''
        i = np.random.randint(self.num_jpegs)
        return self.get(i)
    
class Jstore():
    def __init__(self,main_filename,type_='not-bin'):
        self.main_filename = main_filename
        print 'Loading jstore file...'
        if type_=='not-bin':
            # self.file_, self.keys, self._offset_array = _load_jstore(main_filename)
            self.file_, self._offset_array = _load_jstore(main_filename)
        elif type_=='bin':
            self.file_, self._offset_array = _load_jstore_bin(main_filename)
        self._size_array = np.diff(self._offset_array)
        self.num_jpegs = len(self._size_array)
    
    def get(self,i):
        ''' Jstore.get(i)
        Finds the i-th JPEG in the JSTORE, and returns it as a numpy array.
        '''
        return _read_string(self.file_,self._offset_array[i],self._size_array[i])
    
    def get_rand(self):
        ''' Jstore.get_rand()
        Finds a random JPEG in the JSTORE, and returns it as a numpy array.
        '''
        i = np.random.randint(self.num_jpegs)
        return self.get(i)

def speed_test(jstore,n=100):
    times = list()
    
    for i in xrange(n):
        tic = time()
        jstore.get_rand()
        toc = time()
        times.append(toc-tic)
    
    print 'Mean time (n=%d): %.2e secs.' % (n,np.mean(np.array(times)))
    
# Important function
def make_jstore(main_filename,jpeg_dir):
    sizes = list()
    keys = list()
    
    jstore_filename, keys_filename, offset_filename = _gen_filenames(main_filename)
    
    print 'MAKE_JSTORE'
    print 'Takes a filename, and nested directories of jpegs'
    print 'and makes one large binary file'
    print '-----'
    
    print 'Finding images...'
    jpeg_filenames = tp_utils.ls_r_jpeg2(jpeg_dir)
    perc = np.floor(len(jpeg_filenames)/100.0)
    
    jstore_file = open(jstore_filename,'wb')
    
    print 'Creating %s...' % jstore_filename
    
    tic = time()
    for count, jpeg_filename in enumerate(jpeg_filenames):
        if (count % perc) == 0:
            toc = time()
            print 'Loaded %d%% of images. Took %.2e secs.' % (np.floor(count/perc),toc-tic) 
            tic = time()
        jpeg_string = _jpeg_dump(jpeg_filename)
        jstore_file.write(jpeg_string)
        sizes.append(len(jpeg_string))
        key = jpeg_filename.split('/')[-1].split('.')[0]
        keys.append(key)
    jstore_file.close()
    
    
    size_array = np.array(sizes)
    offset_array = np.concatenate((np.array([0]),np.cumsum(size_array)))
    
    print 'Creating %s...' % keys_filename
    _pickle(keys,keys_filename)
    #cPickle.dump(keys,open(keys_filename,'w'))
    print 'Creating %s...' % offset_filename
    _pickle(offset_array,offset_filename)
    #cPickle.dump(offset_array,open(offset_filename,'w'))
    
    print 'Done.'
    
# Internal helper functions
def _mem_load_jstore(main_filename):
    # jstore_filename, keys_filename, offset_filename = _gen_filenames(main_filename)
    jstore_filename, offset_filename = _gen_filenames(main_filename)
    mem_jstore = _mem_open(jstore_filename)
    # keys = _unpickle(keys_filename)
    offset_array = _unpickle(offset_filename)
    # return mem_jstore, keys, offset_array
    return mem_jstore, offset_array

def _mem_load_jstore_bin(main_filename):
    jstore_filename, header_filename = _gen_filenames_bin(main_filename)
    mem_jstore = _mem_open(jstore_filename)
    offset_array = _read_header(header_filename)
    return mem_jstore, offset_array

# Just used for testing
def _load_jstore(main_filename):
    # jstore_filename, keys_filename, offset_filename = _gen_filenames(main_filename)
    jstore_filename, offset_filename = _gen_filenames(main_filename)
    jstore = open(jstore_filename,'rb')
    # keys = _unpickle(keys_filename)
    offset_array = _unpickle(offset_filename)
    # return jstore, keys, offset_array
    return jstore, offset_array

def _load_jstore_bin(main_filename):
    jstore_filename, header_filename = _gen_filenames_bin(main_filename)
    jstore = open(jstore_filename,'rb')
    offset_array = _read_header(header_filename)
    return jstore, offset_array

def _read_header(header_filename):
    header_file = open(header_filename,'rb')
    header_string = header_file.read()
    header_file.close()
    header = np.fromstring(header_string,dtype=np.uint64)
    num_jpegs = np.uint32(header.shape[0]/5.0)
    header = header.reshape((num_jpegs,5)).T
    offsets = np.concatenate((np.array([0]),header[0,:])).astype(np.uint64)
    return offsets

def _gen_filenames(main_filename):
    jstore_filename = '%s.jstore' % main_filename
    # keys_filename = '%s_keys.pickle' % main_filename
    offset_filename = '%s_offset.pickle' % main_filename
    # return jstore_filename, keys_filename, offset_filename
    return jstore_filename, offset_filename

def _gen_filenames_bin(main_filename):
    jstore_filename = '%s.jstore' % main_filename
    header_filename = '%s_header.bin' % main_filename
    return jstore_filename, header_filename

def _read_string(file,offsets,sizes):
    file.seek(offsets)
    try:
        string = file.read(sizes)
    except:
        print 'offsets: %d' % offsets
        print 'sizes: %d' % sizes
        raise
    try:
        string_file = StringIO(string)
    except:
        print 'StringIO failed.'
    jpeg = Image.open(string_file).convert('RGB')
    return np.array(jpeg)

def _jpeg_dump(filename):
    jpeg_file = open(filename,'rb')
    jpeg_string = jpeg_file.read()
    jpeg_file.close()
    return jpeg_string

def _mem_open(filename):
    file = open(filename,'rb')
    tic = time()
    strFile = StringIO(file.read())
    file.close()
    toc = time()
    print 'MEM_OPEN took %.2e secs.' % (toc-tic)
    return strFile

def _pickle(object_,filename):
    cPickle.dump(object_,open(filename,'w'))
    
def _unpickle(filename):
    object_ = cPickle.load(open(filename,'r'))
    return object_
