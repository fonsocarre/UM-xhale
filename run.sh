#!/bin/env sh

source ~/.bashrc

conda activate sharpy_env
source ./../../../code_xhale/sharpy/bin/sharpy_vars.sh

rm ./*.solver.txt
python generate_xhale.py

for file in ./xhale*.solver.txt; do
    echo "Running " $file
    sharpy $file
done
