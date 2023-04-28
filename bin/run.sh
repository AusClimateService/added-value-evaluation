#!/bin/bash
#PBS -N added_value
#PBS -P tp28
#PBS -q normalbw
#PBS -l walltime=04:00:00
#PBS -l mem=256GB
#PBS -l ncpus=28
#PBS -l storage=gdata/xv83+gdata/dp9+scratch/dp9+gdata/ia39+gdata/rt52+gdata/cj37+gdata/du7+gdata/hh5+gdata/hd50+scratch/hd50+gdata/oi10+gdata/rr8+gdata/rr3+gdata/ma05+gdata/r87+gdata/ub4+gdata/tp28+gdata/du7+gdata/access+gdata/hh5+scratch/du7+scratch/e53+scratch/du7+scratch/access+scratch/public+scratch/tp28+gdata/fs38
#PBS -l jobfs=400GB

module use /g/data/access/ngm/modules
module load analysis3/21.10

# module use /g/data3/hh5/public/modules
# module load conda/analysis3


export PYTHONPATH=/home/565/cst565/ACS_added_value/BARPA_evaluation/lib:$PYTHONPATH
export LIB_LOGLEVEL="INFO"

for gcm in ACCESS-ESM1-5 ACCESS-CM2 EC-Earth3 NorESM2-MM
do
    for variable in tasmax tasmin pr
    do
        echo ======================================================
        echo ${variable}
        echo ======================================================
        python main.py --gcm ${gcm} --rcm BARPA-R --obs AGCD \
                    --scenario-hist historical --scenario-fut ssp370 \
                    --variable ${variable} --freq day \
                    --av-measures realised_added_value \
                    --region Australia --seasons annual JJA DJF SON MAM \
                    --process quantile --process-kwargs '{"quantile":0.999}' --av-distance-measure AVse --pav-distance-measure PAVdiff \
                    --lat0 -45 --lat1 -9 --lon0 110 --lon1 160 \
                    --datestart-hist 19850101 --dateend-hist 20141231 \
                    --datestart-fut 20700101 --dateend-fut 20991231 \
                    --nworkers 4 --nthreads 2 --nprocs 2 \
                    --odir /g/data/tp28/cst565/ACS_added_value \
                    --log-level INFO \
                    --overwrite False
    done
done