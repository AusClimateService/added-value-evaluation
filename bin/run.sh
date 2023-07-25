#!/bin/bash
#PBS -N added_value
#PBS -P tp28
#PBS -q normalbw
#PBS -l walltime=48:00:00
#PBS -l mem=256GB
#PBS -l ncpus=28
#PBS -l storage=gdata/dk92+gdata/zv2+gdata/xv83+gdata/dp9+scratch/dp9+gdata/ia39+gdata/rt52+gdata/cj37+gdata/du7+gdata/hh5+gdata/hd50+scratch/hd50+gdata/oi10+gdata/rr8+gdata/rr3+gdata/ma05+gdata/r87+gdata/ub4+gdata/tp28+gdata/du7+gdata/access+gdata/hh5+scratch/du7+scratch/e53+scratch/du7+scratch/access+scratch/public+scratch/tp28+gdata/fs38
#PBS -l jobfs=400GB

module use /g/data/access/ngm/modules
module load analysis3/21.10

cd /home/565/cst565/ACS/added-value-evaluation/bin

high_quantiles="0.9 0.95 0.99 0.999"
low_quantiles="0.1 0.05 0.01 0.001"
gcm_list="ACCESS-ESM1-5 ACCESS-CM2 EC-Earth3 NorESM2-MM CMCC-ESM2"
# regions="Australia"
regions="Australia Coastline_Australia_100km Topography_Australia_500"
opath="/g/data/tp28/cst565/ACS_added_value"

export PYTHONPATH=/home/565/cst565/ACS/BARPA_evaluation/lib:$PYTHONPATH
export LIB_LOGLEVEL="WARN"
nprocs=2

for gcm in ERA5 $gcm_list
do
    for variable in tasmax tasmin pr
    do
        if [ $variable = "pr" ]; then
            cmd_add=" --agcd_mask"
        else
            cmd_add=""
        fi
        for quantile in $high_quantiles
        do
            kwargs=$(echo {\"quantile\":$quantile})
            echo ======================================================
            echo $(date +"%T") - ${gcm} - ${variable} - ${quantile}
            echo ======================================================
            python main.py --gcm ${gcm} --rcm BARPA-R --obs AGCD \
                        --scenario-hist historical --scenario-fut ssp370 \
                        --variable ${variable} --freq day \
                        --av-measures added_value_norm \
                        --regions ${regions} --seasons annual JJA DJF SON MAM \
                        --process quantile --process-kwargs $kwargs --av-distance-measure AVse --pav-distance-measure PAVdiff \
                        --lat0 -45 --lat1 -9 --lon0 110 --lon1 160 \
                        --datestart-hist 19850101 --dateend-hist 20141231 \
                        --datestart-fut 20700101 --dateend-fut 20991231 \
                        --nworkers 3 --nthreads 1 --nprocs $nprocs \
                        --odir ${opath} \
                        --log-level ${LIB_LOGLEVEL} \
                        --overwrite ${cmd_add}
        done
    done
done

for gcm in ERA5 $gcm_list
do
    for variable in tasmin
    do
        for quantile in $low_quantiles
        do
            kwargs=$(echo {\"quantile\":$quantile})
            echo ======================================================
            echo $(date +"%T") - ${gcm} - ${variable} - ${quantile}
            echo ======================================================
            python main.py --gcm ${gcm} --rcm BARPA-R --obs AGCD \
                        --scenario-hist historical --scenario-fut ssp370 \
                        --variable ${variable} --freq day \
                        --av-measures added_value_norm \
                        --regions ${regions} --seasons annual JJA DJF SON MAM \
                        --process quantile --process-kwargs $kwargs --av-distance-measure AVse --pav-distance-measure PAVdiff \
                        --lat0 -45 --lat1 -9 --lon0 110 --lon1 160 \
                        --datestart-hist 19850101 --dateend-hist 20141231 \
                        --datestart-fut 20700101 --dateend-fut 20991231 \
                        --nworkers 3 --nthreads 1 --nprocs $nprocs \
                        --odir ${opath} \
                        --log-level ${LIB_LOGLEVEL} \
                        --overwrite
        done
    done
done

for gcm in $gcm_list
do
    for variable in tasmax tasmin pr
    do
        if [ $variable = "pr" ]; then
            cmd_add=" --agcd_mask"
        else
            cmd_add=""
        fi
        for quantile in $high_quantiles
        do
            kwargs=$(echo {\"quantile\":$quantile})
            echo ======================================================
            echo $(date +"%T") - ${gcm} - ${variable} - ${quantile}
            echo ======================================================
            python main.py --gcm ${gcm} --rcm BARPA-R --obs AGCD \
                        --scenario-hist historical --scenario-fut ssp370 \
                        --variable ${variable} --freq day \
                        --av-measures realised_added_value \
                        --regions ${regions} --seasons annual JJA DJF SON MAM \
                        --process quantile --process-kwargs $kwargs --av-distance-measure AVse --pav-distance-measure PAVdiff \
                        --lat0 -45 --lat1 -9 --lon0 110 --lon1 160 \
                        --datestart-hist 19850101 --dateend-hist 20141231 \
                        --datestart-fut 20700101 --dateend-fut 20991231 \
                        --nworkers 3 --nthreads 1 --nprocs $nprocs \
                        --odir ${opath} \
                        --log-level ${LIB_LOGLEVEL} \
                        --overwrite ${cmd_add}
        done
    done
done

for gcm in $gcm_list
do
    for variable in tasmin
    do
        for quantile in $low_quantiles
        do
            kwargs=$(echo {\"quantile\":$quantile})
            echo ======================================================
            echo $(date +"%T") - ${gcm} - ${variable} - ${quantile}
            echo ======================================================
            python main.py --gcm ${gcm} --rcm BARPA-R --obs AGCD \
                        --scenario-hist historical --scenario-fut ssp370 \
                        --variable ${variable} --freq day \
                        --av-measures realised_added_value \
                        --regions ${regions} --seasons annual JJA DJF SON MAM \
                        --process quantile --process-kwargs $kwargs --av-distance-measure AVse --pav-distance-measure PAVdiff \
                        --lat0 -45 --lat1 -9 --lon0 110 --lon1 160 \
                        --datestart-hist 19850101 --dateend-hist 20141231 \
                        --datestart-fut 20700101 --dateend-fut 20991231 \
                        --nworkers 3 --nthreads 1 --nprocs $nprocs \
                        --odir ${opath} \
                        --log-level ${LIB_LOGLEVEL} \
                        --overwrite
        done
    done
done