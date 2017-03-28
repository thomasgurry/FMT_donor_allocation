# FMT_donor_allocation
Scripts and software for donor allocation in adaptive FMT trials.  Paper:

Oleson S*, Gurry T*, Alm EJ.  __Statistical Methods in Medical Research__ 2017

## Miscellaneous notes

* Must compile ppp.c with relevant libraries to do Bayesian allocation (see comments for gcc command).

* Make sure to create a Python 3 anaconda environment before running if you don't have this already:

conda create -n py3 python=3 anaconda
source activate py3

* If you get an error of the sort 'error while loading shared libraries: libgsl.so.19: cannot open shared object file: No such file or directory':

LD_LIBRARY_PATH=/usr/local/lib
export LD_LIBRARY_PATH


