#!/bin/bash

LOG_FILE="msm_combined_log.txt"
> "$LOG_FILE"

MIN_EXPONENT=10
MAX_EXPONENT=22
PRECOMP_START=7
PRECOMP_END=24
C_START=9
C_END=23

for exponent in $(seq $MIN_EXPONENT $MAX_EXPONENT); do
    for precomp in $(seq $PRECOMP_START $PRECOMP_END); do
        for c in $(seq $C_START $C_END); do
            ./test_msm $exponent 1 $precomp $c >> "$LOG_FILE" 2>&1
        done
    done
done