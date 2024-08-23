#!/bin/bash

# Define variables
REMOTE_PATH="/work/bm1183/m301049/orcestra"
FLIGHTS=("HALO-20240811a" "HALO-20240813a")
RADIOS=("KV" "183" "11990")

# Check and create directories if they don't exist
for FLIGHT in ${FLIGHTS[@]}; do
    echo "Checking directories for flight: ${FLIGHT}"
    FLIGHT_PATH="${REMOTE_PATH}/${FLIGHT}"
    if [ ! -d "${FLIGHT_PATH}" ]; then
        echo "Directory ${FLIGHT_PATH} does not exist. Creating it."
        mkdir -p "${FLIGHT_PATH}"
        mkdir -p "${FLIGHT_PATH}/bahamas"
        mkdir -p "${FLIGHT_PATH}/radiometer"
        mkdir -p "${FLIGHT_PATH}/radar"
        for RADIO in ${RADIOS[@]}; do
             mkdir -p "${FLIGHT_PATH}/radiometer/${RADIO}"
        done
    else
        echo "Directory ${FLIGHT_PATH} already exists."
    fi
done
