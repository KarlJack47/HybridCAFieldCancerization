#!/bin/bash

if [[ $# -eq 0 ]]; then
  resultsDir="Results"
else
  resultsDir=$1
fi

if [ ! -d "$resultsDir" ]; then
  echo "The results aren't in this directory."
  exit 1
fi

currDir=$(pwd)
cd $resultsDir
for ((i=1; i<38; i++)); do
  if [ ! -d "$i"_* ]; then
    continue
  fi

  if [[ $i -eq 1 ]] || [[ $i -eq 3 ]]; then
    numCells=$((128*128))
  elif [[ $i -eq 2 ]]; then
    numCells=$((64*64))
  elif [[ $i -eq 5 ]]; then
    numCells=$((512*512))
  else
    numCells=$((256*256))
  fi

  if [[ $i -eq 1 ]]; then
    nCarcin=0
    carcins=()
  elif [[ $i -eq 9 ]] || [[ $i -eq 10 ]] || [[ $i -eq 12 ]] || [[ $i -eq 13 ]]; then
    nCarcin=1
    if [[ $i -eq 9 ]]; then
      carcins=(0)
    else
      carcins=(1)
    fi
  else
    nCarcin=2
    carcins=(0 1)
  fi

  echo $i
  cd "$i"_*/1/data/data_files && gnuplot -c $currDir/bin/create_plots $numCells $nCarcin ${carcins[@]} 1 > /dev/null 2>&1 && mv *.png ../../graphs && cd $currDir/$resultsDir
done

exit 0