# tp_detect_tools.py
import pickle
from time import time

import numpy as np
from scipy import ndimage
from scipy import misc

import tp_tools as tp

def vectorize(img):
    """
    Vectorize an rgb image, one channel at a time
    """
    r0 = img[:,:,0].flatten()
    r1 = img[:,:,1].flatten()
    r2 = img[:,:,2].flatten()
    r3 = img[:,:,3].flatten()
    r4 = img[:,:,4].flatten()
    r5 = img[:,:,5].flatten()
    r6 = img[:,:,6].flatten()
    r7 = img[:,:,7].flatten()
    r8 = img[:,:,8].flatten()
    r9 = img[:,:,9].flatten()
    r10 = img[:,:,10].flatten()
    r11 = img[:,:,11].flatten()
    r12 = img[:,:,12].flatten()
    r13 = img[:,:,13].flatten()
    r14 = img[:,:,14].flatten()
    r15 = img[:,:,15].flatten()
    vect = np.concatenate((r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15))
    return vect

def gen_object_index(annotations):
	object_index = []
	for annotation in annotations:
		object_set = set([object['name'] for object in annotation])
		object_index.append(list(object_set))
	return object_index

def get_unique_objects(index):
	object_list = []
	for item in index:
		object_list = object_list + item

	return tp.natural_sort(list(set(object_list)))

def random_crop(image_size, width_x, width_y):
	space_y = image_size[0]-width_y
	space_x = image_size[1]-width_x

	x = np.random.randint(space_x)
	y = np.random.randint(space_y)

	crop = {'x': x,
			'y': y,
			'width_x': width_x,
			'width_y': width_y,
			}
	return crop

class LocationMaskStore():
	def __init__(self, filename, mask_size=24):
		self.filename = filename
		self.mask_size = mask_size

		images_info = pickle.load(open('%s.pickle' % filename,'rb'))
		annotations = [image_info['objects'] for image_info in images_info]
		object_index = gen_object_index(annotations)
		object_names = get_unique_objects(object_index)

		object_dict = {}
		for i, name in enumerate(object_names):
			object_dict[name]=i

		self.annotations = annotations
		self.images_info = images_info
		self.object_index = object_index
		self.object_names = object_names
		self.object_dict = object_dict

	def search(self, class_id):
		temp = search(self.object_index, self.object_names[class_id])
		not_temp = [not item for item in temp]
		hits = np.where(temp)[0]
		misses = np.where(not_temp)[0]
		return hits, misses

	def get(self, ind, crop=None):

		height = self.images_info[ind]['height']
		width  = self.images_info[ind]['width']

		crop_max = np.min((height,width))
		crop_min = np.int(crop_max*0.8)
		crop_size = np.random.randint(crop_min,crop_max)

		crop = random_crop((height,width),crop_size,crop_size)

		mask = np.zeros((29,29,16))
		for item in self.annotations[ind]:
			id = self.object_dict[item['name']]
			if id < 16:
				mask_temp = np.zeros((height,width))
				xmax = item['xmax']
				xmin = item['xmin']
				ymax = item['ymax']
				ymin = item['ymin']
				mask_temp[ymin:ymax,xmin:xmax] = 1
                                mask_temp = mask_temp[crop['y']:crop['y']+crop['width_y'],crop['x']:crop['x']+crop['width_x']]
                                mask_temp = misc.imresize(mask_temp,(29,29)).astype(np.float32)/255.0
				mask[:,:,id] = mask_temp

		#mask = mask[crop['y']:crop['y']+crop['width_y'],crop['x']:crop['x']+crop['width_x']]
                
		#mask = ndimage.interpolation.zoom(mask,(29.5/crop['width_y'],29.5/crop['width_x'],1))

		return mask, crop
