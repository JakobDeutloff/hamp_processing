#!/bin/bash

# Define variables
LOCAL_PATH="/Volumes/ORCESTRA"
REMOTE_SSH="levante.dkrz.de"
REMOTE_PATH="/work/bm1183/m301049/orcestra"
FLIGHTS=("HALO-20240811a" "HALO-20240813a")
RADIOS=("KV" "183" "11990")

# Copy data for all flights
for FLIGHT in ${FLIGHTS[@]}; do
    echo "Transfering data for flight: ${FLIGHT}"
    # Bahamas data
    rsync -av --ignore-existing ${LOCAL_PATH}/${FLIGHT}/bahamas/*.nc ${REMOTE_SSH}:${REMOTE_PATH}/${FLIGHT}/bahamas/
    # radiometer data
    for RADIO in ${RADIOS[@]}; do
        rsync -av --ignore-existing ${LOCAL_PATH}/${FLIGHT}/radiometer/${RADIO}/${FLIGHT:7:6}.BRT.NC ${REMOTE_SSH}:${REMOTE_PATH}/${FLIGHT}/radiometer/${RADIO}/
        rsync -av --ignore-existing ${LOCAL_PATH}/${FLIGHT}/radiometer/${RADIO}/${FLIGHT:7:6}.IWV.NC ${REMOTE_SSH}:${REMOTE_PATH}/${FLIGHT}/radiometer/${RADIO}/
    done
done
