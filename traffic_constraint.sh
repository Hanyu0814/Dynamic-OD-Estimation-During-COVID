#!/bin/bash

module load gcc
module load python/3.7

start_m=$1
start_d=$2
end_m=$3
end_d=$4
year=$5
flag_weekday=$6
beta=$7
gamma=$8

out_put_file="Output.out"

echo $start_m , $start_d , $end_m , $end_d

day_start="$year"+$start_m+$start_d
day_end="$year"+$end_m+$end_d

echo $day_start , $day_end

mkdir -p "R_matrix"
mkdir -p "P_matrix"
mkdir -p "X_vector"
mkdir -p "Q_vector"
mkdir -p "Constraints"

python3 Traffic_constraint.py "${start_m}" "${start_d}" "${end_m}" "${end_d}" "${year}" "${flag_weekday}" "${beta}" "${gamma}"
