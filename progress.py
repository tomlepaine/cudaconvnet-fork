# progress.py

from time import time
import numpy as np

class progress():
	def __init__(self, num, name='Progress',verbose=True):
		self.name = name
		self.verbose = verbose
		self.ptic = time()
		self.tic = time()
		self.perc = round(num/100.0)

	def update(self,count):
		if self.verbose:
			if ((count+1) % self.perc) == 0:
				toc = time()
				print '%s: %d%% in %.2e secs.' % (self.name, np.floor(count/self.perc),toc-self.tic)
				self.tic = time()

	def end(self):
		toc = time()
		print 'Done. Elapsed time %.2e secs.' % (toc-self.ptic)