#!/bin/sh

#cc -o ehw.o ehw_new.c -lm -lfftw3 -Wextra -fopenmp -lnetcdf
mpicc -o ehw.o ehw_new.c -lm -lfftw3 -Wextra -fopenmp -lnetcdf

#./ehw.o 40 256 5 1.0
mpirun -np 4 ./ehw.o 40 256 5 1.0
