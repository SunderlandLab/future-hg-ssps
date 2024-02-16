#!/bin/bash
analysis_yr_lo="2016"
analysis_yr_hi="2019"
#emission_yr="2020"

echo "Bash version ${BASH_VERSION}..."

BASE_PATH="/Volumes/Expansion/Streets_Projections/Model_Output/Streets_Future/"
declare -a StringArray=("SSP1-26" "SSP2-45" "SSP5-34" "SSP5-85" )

for EMISSION_YR in 2020 2030 2040 2050 2060 2070 2080 2090 2100
do
  for SCENARIO_NAME in ${StringArray[@]}
  do
    path="${BASE_PATH}${SCENARIO_NAME}_${EMISSION_YR}"
    echo $path
    python process_raw_GC_data.py $path "${analysis_yr_lo}-01-16T00:00:00.000000000", "${analysis_yr_hi}-1-16T00:00:00.000000000"
  done
done

# now call for baseline condition
path="${BASE_PATH}baseline"
echo $path
python process_raw_GC_data.py $path "${analysis_yr_lo}-01-16T00:00:00.000000000", "${analysis_yr_hi}-1-16T00:00:00.000000000"