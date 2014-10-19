convnet-asgd
============
An extension of Alex Krizhevsky's [cuda-convnet][cudaconv] code. 

Changes:
+ Adds A-SGD functionality.
+ Works with cuda 5.0.
+ Added dropout.
+ Has dataproviders for imagenet.

The A-SGD additions happen in two files:
+ `asgd.py`
+ `modelcomm.py`

## Installation
Requirements differences with `cuda-convnet`:
- Cuda 4.2
+ Cuda 5.0
+ MPI
+ [mpi4py][mpi4py]

To make this run on a cluster such as `guillimin` you might need to load a version of mpi. Here is an example for openmpi.
Load a version of openmpi like so:
`module load openmpi/1.6.3-gcc`

Set the mpi dir in mpi4py's `mpi.cfg` file like so:
```
[openmpi]
mpi_dir              = /software/CentOS-5/tools/openmpi-1.6.3-gcc
```

Then build mpi like so:
```
python setup.py build --mpi=openmpi
python setup.py install --home=~
```

[cudaconv]: https://code.google.com/p/cuda-convnet/
[mpi4py]: http://mpi4py.scipy.org/docs/usrman/index.html

## Running
The package is designed to be run just like `cuda-convnet`. It takes the same arguments but instead of calling `python convnet.py` call `python asgd.py`.
But it should be called with `mpiexec` with the correct number of nodes. For instance if we 
wanted 16 clients (plus 1 server) we would call the `asgd.py` like so:
```
mpiexec -n 17 -N 1 python asgd.py \
--data-path=/projects/sciteam/joi/data/ \
--save-path=/projects/sciteam/joi/save/mpi/double/16 \
--test-range=1-390 \
--train-range=1-10009 \
--layer-def=./good-layers/double-layers5.cfg \
--layer-params=./good-layers/double-params3.cfg \
--data-provider=imagenetJstore \
--test-one=1 \
--test-freq=600 \
```
