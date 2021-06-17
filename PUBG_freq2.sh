#!/bin/tcsh
##
##	Job Script for PC Cluster, JAIST
##		created by mkjob.pl
##	** Revise the script as necessary **
#
#PBS -N PUBG_heatmap
#PBS -q LONG 
#PBS -j oe 
#PBS -l select=4:ncpus=8

cd ${PBS_O_WORKDIR}

setenv PATH /home/$USER/anaconda3/bin:${PATH}
setenv PYTHON_PATH /home/$USER/anaconda3/bin/python

cd ~/PUBG_freq3/
python heatmap_plt.py 

