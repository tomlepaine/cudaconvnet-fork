convnet
============
Simple extension of Alex Krizhevsky's [cuda-convnet][cudaconv] code. 

Changes:
+ Works with cuda 5.0.
+ Added dropout.
+ Has dataproviders for imagenet.

The dataprovider extensions happen over too many files:
convdata2.py
jstore.py
tp_utils.py

## Installation
Requirements differences with `cuda-convnet`:
- Cuda 4.2
+ Cuda 5.0