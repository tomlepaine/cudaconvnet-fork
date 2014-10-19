# tp_image_tools.py

# Standard Library
import os
from StringIO import StringIO

# Third party
import Image
import numpy as np
import scipy.misc as misc

def _read_meta(filename):
    sizes = list()
    xs = list()
    ys = list()
    filenames = list()

    f = open(filename,'r')

    for line in f:
        data = line.split('\n')[0].split(', ')
        sizes.append(np.int64(data[0]))
        xs.append(np.int(data[1]))
        ys.append(np.int(data[2]))
        filenames.append(data[3])
    
    return sizes, xs, ys, filenames

def _test_string(string):
    img = np.array(Image.open(StringIO(string)))
    plt.imshow(img)
    plt.show()
    
def _test_img(img):
    plt.imshow(img)
    plt.show()
    
class Jstore():
    def __init__(self, filename):
        
        bin_name = '%s.jstore_bin' % filename
        meta_name = '%s.jstore_meta' % filename

        if not os.path.exists(bin_name):
            raise Exception('Jstore bin does not exist.')
        if not os.path.exists(meta_name):
            raise Exception('Jstore meta does not exist.')
        
        # Load meta data
        sizes, xs, ys, filenames = _read_meta(meta_name)

        # Calculate offsets
        temp_size = np.concatenate((np.zeros(1,dtype=np.int64),np.array(sizes)))
        offsets = list(np.cumsum(temp_size))
        
        f = open(bin_name,'rb')
        
        # Public APIs
        self.bin = bin_name
        self.meta = meta_name
        self.num_files = len(filenames)
        
        # Private APIs
        self._sizes = sizes
        self._offsets = offsets
        self._xs = xs
        self._ys = ys
        self._filenames = filenames
        self._bin_file = f
        
    def _read_string(self,ind):
        f = self._bin_file
        offsets = self._offsets
        sizes = self._sizes
        
        f.seek(offsets[ind])
        string = f.read(sizes[ind])
        return string
    
    def get(self, ind, crop=None):
        """
        Returns img #IND
        """
        string = self._read_string(ind)
        temp_img = np.array(Image.open(StringIO(string)).convert('RGB'))
	if crop:
            x = crop['x']
            y = crop['y']
            win_x = crop['width_x']
            win_y = crop['width_y']
            img = temp_img[y:y+win_y, x:x+win_x, :]
        else:
            img = temp_img
	
	#print img.shape
        img = misc.imresize(img,(224,224)).astype(np.float32)
        return img
    
    def get_size(self, ind):
        return self._xs[ind], self._ys[ind]
    
    def get_filename(self, ind):
        return self._filenames[ind]
    
def gen_centers(x,y,win,batch_size):
    """
    Generates BATCH_SIZE random crops
    Making sure they fit an image of size (X,Y)
    Returns the centers XS, YS
    """

    if win>x or win>y:
        raise Exception('Crop is bigger than the image.')

    x_bound, y_bound = x-win+1, y-win+1
    xs = np.random.randint(0,x_bound,batch_size)+win/2
    ys = np.random.randint(0,y_bound,batch_size)+win/2
    return xs, ys

def write_crops(batch_num, name, xs, ys,win):
    """
    Saves a csv file named 'BATCH_NUM_NAME_crops.csv'
    Where line i is:
    i, x_i, y_i, win
    """
    filename = '%d_%s_crops.csv' % (batch_num, name.split('.JPEG')[0])
    f = open(filename,'w')
    count = 0
    for x, y in zip(xs,ys):
        temp = '%d, %d, %d, %d\n' % (count, x, y, win)
        f.write(temp)
        count += 1
    f.close()
        
def get_crops(image, xs, ys, win):
    """
    Returns a matrix DATA
    The columns of DATA are crops of IMAGE
    Centered at x_i, y_i, and of size (win x win)
    """
    batch_size = len(xs)
    data = np.ones((batch_size,win*win*3),dtype=np.single)
    count = 0
    for x, y in zip(xs,ys):
        xmin = x-win/2
        xmax = x+win/2
        ymin = y-win/2
        ymax = y+win/2
        temp = image[xmin:xmax,ymin:ymax,:]
        temp2 = misc.imresize(temp,(224,224))
        data[count,:] = vectorize(temp2)
        count += 1
    return np.transpose(data)

def vectorize(img):
    """
    Vectorize an rgb image, one channel at a time
    """
    r = img[:,:,0].flatten()
    g = img[:,:,1].flatten()
    b = img[:,:,2].flatten()
    vect = np.concatenate((r,g,b))
    return vect

def load_average(filename):
    average = np.load(filename).astype(np.float32)
    return vectorize(average[16:256-16,16:256-16,:])

def rescale(image,win):
    x,y,z = image.shape
    if win>x or win>y:
        dim_min = np.min([x,y])*1.0
        new_x = int(round(x/dim_min*win))
        new_y = int(round(y/dim_min*win))
        new_image = misc.imresize(image,(new_x,new_y))
        return new_image, new_x, new_y
    else:
        return image, x, y
