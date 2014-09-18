dump2h5
=======

Convert [nc_dump](https://github.com/peterkuma/nc_dump)
files to NetCDF/HDF5 data files.

Dump is a binary file containing raw data values, accompanied by a set of
metadata files describing data dimensions, type, and attributes.

Installation
------------

Requirements:

* [HDF5](http://www.hdfgroup.org/HDF5/) (C library)
* [netcdf](http://www.unidata.ucar.edu/software/netcdf/) (C library)

On Ubuntu/Debian derived operating systems, install requirements with:

    apt-get install libhdf5-dev libnetcdf-dev

Compile dump2h5 with:

    h5cc -o dump2h5 -Wall main.c

Usage
-----

To import dataset `dataset` into `mydata.nc`, do:

    dump2h5 -o mydata.nc dataset

This requires files `dataset`, `dataset.dims` and `dataset.dtype` to be present
(see description of dump format below). Any existing file `mydata.nc`
will be truncated before import.

To append to an existing output file, use the -a switch:

    dump2h5 -a -o mydata.nc dump

To import all datasets in directory `dump` into `mydata.nc`, do:

    dump2h5 -o mydata.nc dump

Note that subdirectories are not traversed.

Format of dump files
--------------------

A dump consists of the following files:

* `<dataset>` – raw data
* `<dataset>.dim`  – data dimensions
* `<dataset>.dtype` – data type

where `<dataset>` is an arbitrary dataset name supported by HDF5.

`<dataset>.dim` is a file containing size of each data dimension on a seperate
line, for example:

    1000
    50

for a dataset of size 1000×50. The size of the first dimension can be -1
to signify an unlimited dimension, in which case the size will be computed 
automatically from the raw data file size and the product of subsequent
dimensions.

`<dataset>.dtype` is a file containing a string determining the data type.
Data type can be one of:

* `float32`
* `float64`

File `<dataset>` contains binary raw data packed without padding. Big endian
byte order is assumed.
