# tp_tools.py

# Standard Library
import os
import re

# Third party
import numpy as np

# Found here: http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
# Read more here: 
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def ls_jpeg(dir_path,verbose=True,search=None):
    ''' ls_jpeg(dir_path)
    Recursively lists JPEG files.
    I've only tried it with the imagenet training, validation, and test directories.
    '''
    jpeg_list = list()
    if verbose: print 'Crawling directories...'
    for r,d,files in os.walk(dir_path):
        for file in files:
            if file.endswith('.JPEG') or file.endswith('.jpg'):
                if search:
                    if search in file.split('.')[0].split('/')[-1]:
                        jpeg_list.append(os.path.abspath('%s/%s' %(r,file)))
                else:
                    jpeg_list.append(os.path.abspath('%s/%s' %(r,file)))

    jpeg_list = np.array(natural_sort(jpeg_list))
        
    if verbose: print 'Found %d jpeg files' % len(jpeg_list)
    return jpeg_list

def batches_from_list(image_list,imagesInBatch):
    batches = []
    numBatches = len(image_list)/imagesInBatch
    for i in xrange(0,numBatches):
        start = imagesInBatch*(i)
        end = imagesInBatch*(i+1)
        batches.append(image_list[start:end])
    return batches

def _search(dir='.',format='JPEG'):
	''' Tool Tip '''

	out_files = []
	for root, dirs, files in os.walk(dir):
		for file in files:
			if file.endswith('.%s' % format):
				out_files.append(os.path.join(root,file))
	return out_files

def _hide():
	in_path  = '/home/paine/research/ILSVRC2012_img_train_untar'
	out_path = '/home/paine/research/ILSVRC2012_img_train_crop'

	in_list = search(dir=in_path)
	out_list = [os.path.join(out_path,file.split(path)[-1][1::]) for file in in_list]
