#!/bin/bash

FLIGHT="HALO-20241105a"

LOCAL_PATH="/Users/jakobdeutloff/Programming/Orcestra/hamp_processing/Data"
REMOTE_SSH="m301049@login2.mpimet.mpg.de:/pool/OBS/transfer-LH/HALO"
RADIOMETERS=("KV" "183" "11990")

# create directories
mkdir -p ${LOCAL_PATH}/Radar_Data/${FLIGHT}
for RADIO in ${RADIOMETERS[@]}; do
    mkdir -p ${LOCAL_PATH}/Radiometer_Data/${FLIGHT}/${RADIO}
done


# copy radar data from remote to local
rsync -av --ignore-existing ${REMOTE_SSH}/radar/${FLIGHT}/mom/*.nc ${LOCAL_PATH}/Radar_Data/${FLIGHT}/

# copy radiometer data from remote to local
for RADIO in ${RADIOMETERS[@]}; do
    rsync -av --ignore-existing ${REMOTE_SSH}/radiometer/${FLIGHT}/${RADIO}/${FLIGHT:7:6}.BRT.NC ${LOCAL_PATH}/Radiometer_Data/${FLIGHT}/${RADIO}/

    rsync -av --ignore-existing ${REMOTE_SSH}/radiometer/${FLIGHT}/${RADIO}/${FLIGHT:7:6}.IWV.NC ${LOCAL_PATH}/Radiometer_Data/${FLIGHT}/${RADIO}/
done
