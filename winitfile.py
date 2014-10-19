import util

def weight_from_checkpoint(path, name, idx, shape):
	checkpoint = util.unpickle(path)
	layers = checkpoint['model_state']['layers']
	layer_names = [layer['name'] for layer in layers]
	match = [i for i, layer_name in enumerate(layer_names) if layer_name==name]
	if len(match)>1:
		raise Exception('More than one matching layer found.')
	if len(match)<1:
		raise Exception('No matching layer found.')
	weights = layers[match[0]]['weights'][idx]
	if weights.shape != shape:
		print 'Shape mismatch:'
		print 'yours:', shape
		print 'mine:', weights.shape
	return weights

def bias_from_checkpoint(path, name, shape):
	checkpoint = util.unpickle(path)
	layers = checkpoint['model_state']['layers']
	layer_names = [layer['name'] for layer in layers]
	match = [i for i, layer_name in enumerate(layer_names) if layer_name==name]
	if len(match)>1:
		raise Exception('More than one matching layer found.')
	if len(match)<1:
		raise Exception('No matching layer found.')
	biases = layers[match[0]]['biases']
	if biases.shape != shape:
		print 'Shape mismatch:'
		print 'yours:', shape
		print 'mine:', biases.shape
	return biases

def makew(name, idx, shape, params=None):
	path = '/projects/sciteam/joi/save/best/good_check'
	return weight_from_checkpoint(path, name, idx, shape)

def makeb(name, shape, params=None):
	path = '/projects/sciteam/joi/save/best/good_check'
	return bias_from_checkpoint(path, name, shape)
