#!/bin/bash
analysis_yr_lo="2016"
analysis_yr_hi="2019"

echo "Bash version ${BASH_VERSION}..."

BASE_PATH="/Volumes/Expansion/Streets_Projections/Model_Output/Streets_Future_IFs/"
declare -a StringArray=("AfricaMidEast" "Asia" "Canada" "CentralAm" "Europe" "FmrUSSR" "Oceania" "SouthAm" "USA" )
declare -a StringArrayB=("Hg0" "Hg2" )

for REGION_NAME in ${StringArray[@]}
do
  for SPECIES_NAME in ${StringArrayB[@]}
  do
    path="${BASE_PATH}${REGION_NAME}_${SPECIES_NAME}"
    echo $path
    python process_raw_GC_data.py $path "${analysis_yr_lo}-01-16T00:00:00.000000000", "${analysis_yr_hi}-1-16T00:00:00.000000000"
  done
done

path="${BASE_PATH}blank"
python process_raw_GC_data.py $path "${analysis_yr_lo}-01-16T00:00:00.000000000", "${analysis_yr_hi}-1-16T00:00:00.000000000"

path="${BASE_PATH}LAND_Hg0"
python process_raw_GC_data.py $path "${analysis_yr_lo}-01-16T00:00:00.000000000", "${analysis_yr_hi}-1-16T00:00:00.000000000"

path="${BASE_PATH}OCEAN_Hg0"
python process_raw_GC_data.py $path "${analysis_yr_lo}-01-16T00:00:00.000000000", "${analysis_yr_hi}-1-16T00:00:00.000000000"

path="${BASE_PATH}UNIFORM_Hg0"
python process_raw_GC_data.py $path "${analysis_yr_lo}-01-16T00:00:00.000000000", "${analysis_yr_hi}-1-16T00:00:00.000000000"

path="${BASE_PATH}PERMAFROST_Hg0"
python process_raw_GC_data.py $path "${analysis_yr_lo}-01-16T00:00:00.000000000", "${analysis_yr_hi}-1-16T00:00:00.000000000"