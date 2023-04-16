#!/bin/bash
#PBS -N added_value
#PBS -P tp28
#PBS -q normalbw
#PBS -l walltime=04:00:00
#PBS -l mem=256GB
#PBS -l ncpus=28
#PBS -l storage=gdata/xv83+gdata/dp9+scratch/dp9+gdata/ia39+gdata/rt52+gdata/cj37+gdata/du7+gdata/hh5+gdata/hd50+scratch/hd50+gdata/oi10+gdata/rr8+gdata/rr3+gdata/ma05+gdata/r87+gdata/ub4+gdata/tp28+gdata/du7+gdata/access+gdata/hh5+scratch/du7+scratch/e53+scratch/du7+scratch/access+scratch/public+scratch/tp28+gdata/fs38
#PBS -l jobfs=400GB

module use /g/data3/hh5/public/modules
module load conda/analysis3-unstable

export PYTHONPATH=/home/565/cst565/ACS_added_value/BARPA_evaluation/lib:$PYTHONPATH
export LIB_LOGLEVEL="DEBUG"

python main.py --gcm ACCESS-CM2 --rcm BARPA-R --obs AGCD \
            --scenario-hist historical --scenario-fut ssp370 \
            --variable tasmax --freq day \
            --av-measures realised_added_value \
            --region land --seasons annual JJA DJF SON MAM \
            --process quantile --process-kwargs '{"quantile":0.999}' --av-distance-measure AVse --pav-distance-measure PAVdiff \
            --lat0 -44.46 --lat1 -10.47 --lon0 112.91 --lon1 154.316 \
            --datestart-hist 19850101 --dateend-hist 20141231 \
            --datestart-fut 20700101 --dateend-fut 20991231 \
            --nworkers 4 --nthreads 1 --nprocs 7 \
            --odir /g/data/tp28/cst565/ACS_added_value \
            --log-level DEBUG \
            --overwrite False