#!/bin/bash
#PBS -N addition
#PBS -l nodes=1:ppn=1:gpus=1

cd $PBS_O_WORKDIR

./a.out 
