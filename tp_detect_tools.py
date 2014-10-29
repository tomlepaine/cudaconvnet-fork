# tp_detect_tools.py
import numpy as np
import pickle
import tp_tools as tp
import scipy.misc as misc

class LocationMaskStore():
	def __init__(self, filename, mask_size=24):
		self.filename = filename
		self.mask_size = mask_size

		images_info = pickle.load(open('%s.pickle' % filename,'rb'))
		annotations = [image_info['objects'] for image_info in images_info]
		object_index = gen_object_index(annotations)

		object_names = get_unique_objects(object_index)

		self.annotations = annotations
		self.images_info = images_info
		self.object_index = object_index
		self.object_names = object_names

	def search(self, class_id):
		temp = search(self.object_index, self.object_names[class_id])
		not_temp = [not item for item in temp]
		hits = np.where(temp)[0]
		misses = np.where(not_temp)[0]
		return hits, misses

	def get(self, ind, crop=None, class_id=None):

		height = self.images_info[ind]['height']
		width  = self.images_info[ind]['width']

		crop_max = np.min((height,width))
		crop_min = crop_max/2
		crop_size = np.random.randint(crop_min,crop_max)

		# if class hit
		if self.object_names[class_id] in self.object_index[ind]:
			# generate a good crop
			# make the right mask

			objects = [object for object in self.annotations[ind] if object['name']==self.object_names[class_id]]
			mask = np.zeros((height,width))

			num_objects = len(objects)
			pick = np.random.randint(num_objects)
			i = objects[pick]

			crop = box_crop(i,(height,width),crop_max)
			#plt.imshow(crop_img((height,width),crop)-mask)
			#plt.show()

			for i in objects:
				xmax = i['xmax']
				xmin = i['xmin']
				ymax = i['ymax']
				ymin = i['ymin']
			 	mask[ymin:ymax,xmin:xmax] = 1

			location_mask = mask[crop['y']:crop['y']+crop['width_y'],crop['x']:crop['x']+crop['width_x']]
			location_mask = misc.imresize(location_mask,(self.mask_size,self.mask_size))
		else:
			# generate a random crop
			# make a zeros mask
			crop = random_crop((height,width),crop_size,crop_size)
			location_mask = np.zeros((self.mask_size,self.mask_size))

		location_mask = location_mask.astype(np.float32)/255.0
		return location_mask, crop

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

def search(index, term):
	return [(term in item) for item in index]

def crop_img(image_size, crop):
	mask = np.zeros(image_size)
	mask[crop['y']:crop['y']+crop['width_y'],crop['x']:crop['x']+crop['width_x']] = 1
	return mask

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

def box_crop(object,image_size,crop_max):
	xmin = object['xmin']
	xmax = object['xmax']
	ymin = object['ymin']
	ymax = object['ymax']

	object_size = np.max((xmax-xmin,ymax-ymin))
	
	if object_size>=crop_max:
		crop_min = crop_max/2
	else:
		crop_min = np.max((crop_max/2,object_size))

	crop_size = np.random.randint(crop_min,crop_max)

	height = image_size[0]
	width = image_size[1]

	right = np.argmin((xmin,width-xmax))
	#horizontal_space = np.min((xmin,width-xmax))
	top = np.argmin((height-ymax,ymin))
	#vertical_space = np.min((height-ymax,ymin))

	left_offset = crop_size
	top_offset = crop_size
	right_offset = width-crop_size
	bottom_offset = height-crop_size

	if right:
		ref_x = np.max((left_offset,xmax))
		#print 'ref_x: %s' % ref_x
		#print 'width: %s' % width
		space_x = width - ref_x
		if space_x==0:
			offset_x = 0
		else:
			offset_x = np.random.randint(space_x)
		x = ref_x+offset_x-crop_size
	else: # left
		ref_x = np.min((right_offset,xmin))
		space_x = ref_x
		if space_x==0:
			offset_x = 0
		else:
			offset_x = np.random.randint(space_x)
		x = ref_x-offset_x
		

	if top:
		ref_y = np.min((bottom_offset,ymin))
		space_y = ref_y
		if space_y==0:
			offset_y = 0
		else:
			offset_y = np.random.randint(space_y)
		y = ref_y-offset_y
	else: # bottom
		ref_y = np.max((top_offset,ymax))
		#print 'ref_y: %s' % ref_y
		#print 'height: %s' % height
		space_y = height - ref_y
		if space_y==0:
			offset_y = 0
		else:
			offset_y = np.random.randint(space_y)
		y = ref_y+offset_y-crop_size

	crop = {'x': x,
			'y': y,
			'width_x': crop_size,
			'width_y': crop_size,
			}

	return crop
