#!/bin/bash

FLIGHT="HALO-20241119a"

LOCAL_PATH="/Users/jakobdeutloff/Programming/Orcestra/hamp_processing/Data"
REMOTE_SSH="m301049@login2.mpimet.mpg.de:/pool/OBS/ORCESTRA/raw/HALO"
RADIOMETERS=("KV" "183" "11990")
PASSWORD="Anvilclouds_2026"  # Replace with your actual password

# create directories
mkdir -p ${LOCAL_PATH}/Radar_Data/${FLIGHT}
for RADIO in ${RADIOMETERS[@]}; do
    mkdir -p ${LOCAL_PATH}/Radiometer_Data/${FLIGHT}/${RADIO}
done

# copy radar data from remote to local
sshpass -p "${PASSWORD}" rsync -av --ignore-existing ${REMOTE_SSH}/radar/${FLIGHT}/mom/*.nc ${LOCAL_PATH}/Radar_Data/${FLIGHT}/

# copy radiometer data from remote to local
for RADIO in ${RADIOMETERS[@]}; do
    sshpass -p "${PASSWORD}" rsync -av --ignore-existing ${REMOTE_SSH}/radiometer/${FLIGHT}/${RADIO}/${FLIGHT:7:6}.BRT.NC ${LOCAL_PATH}/Radiometer_Data/${FLIGHT}/${RADIO}/
    if [ $RADIO == "KV" ]; then
        sshpass -p "${PASSWORD}" rsync -av --ignore-existing ${REMOTE_SSH}/radiometer/${FLIGHT}/${RADIO}/${FLIGHT:7:6}.IWV.NC ${LOCAL_PATH}/Radiometer_Data/${FLIGHT}/${RADIO}/
    fi
done
