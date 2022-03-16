#!/bin/bash

if [ "$1" == "help" ]; then
  echo 'Usage:' $0 '[OPTION]'
  echo 'Run the CA model a certain number of times with a set grid size and number of time steps.'
  echo ''
  echo 'Options:'
  echo ' help       prints out the information you are currently reading'
  echo '   -v       flag used to display progress of the script, default=disabled'
  echo '   -f dir   specify the directory you want to save the files in'
  echo '   -d       enables the gui, default=disabled'
  echo '   -s       enables saving of pictures and videos, default=disabled'
  echo '   -C       enables saving the state of each cell in a compressed file'
  echo '   -n int   number of simulations to run, default=10'
  echo '   -t int   number of time steps, default=8766'
  echo '   -g int   grid size, default=256 (power of 2 between 16 and 1024)'
  echo '   -i int   init type, default=0, 0=one carcinogen,'
  echo '            1=start with tumour cells, 2=two carcinogen, 3=no carcinogen'
  echo '   -k       enables programming pausing on first TC being formed, default=disabled'
  echo '   -j int   number of time steps until excision is performed or prompted to perform, default=-1'
  echo '   -p       enables perfect tumour excision, default=disabled'
  echo '   -q       enables removal of the field during perfect excision mode, default=disabled'
  echo '   -u int   number neighbourhoods around a cell to include in the excision for the perfect'
  echo '            excision mode'
  echo '   -e int   frequency of tumour excision, default=-1 meaning no excision'
  echo '            is performed unless excision mode is activated in the gui'
  echo '   -c int   activate specified carcinogen'
  echo '   -h int   specifies which carcinogen function to use'
  echo '   -o int   carcinogen function type 0=carcinogen function, 1=PDE,'
  echo '            2=carcinogen function multiplied by PDE'
  echo '   -x int   specify how long exposure of carcinogen is when it is being periodicly'
  echo '            activated and deactivated'
  echo '   -a int   number of time steps with carcinogen'
  echo '   -b int   number of time steps without carcinogen'
  echo '   -m int   number of positively mutated genes required for a cell to be considered mutated'
  echo '   -r float chance a CSC forms'
  exit 0
fi

if [ ! -f main ]; then
  echo 'The main program is not within the current directory.'
  exit 1
fi
currDir=$(pwd)

verbose=0
numSim=10
options=()
carcinogens=()
nCarcin=0
gridSize=256
outFolder=output_$(date -d "today" +"%Y%m%d%H%M%S")
while getopts vf:dsCn:t:g:i:c:a:b:x:h:o:m:r:kj:pqu:e: option; do
  case "${option}" in
    v) verbose=1;;
    f) outFolder=${OPTARG};;
    d) options+=("-d");;
    s) options+=("-s");;
    C) options+=("-C");;
	n) options+=("-n ${OPTARG}")
       numSim=${OPTARG}
       ;;
    t) options+=("-t ${OPTARG}");;
    g) options+=("-g ${OPTARG}")
       gridSize=${OPTARG}
       ;;
    i) options+=("-i ${OPTARG}")
       if [ ${OPTARG} -eq 0 ]; then
         nCarcin=1
       fi
       if [ ${OPTARG} -eq 2 ]; then
         nCarcin=2
         carcinogens=(0 1)
       fi;;
    c) options+=("-c ${OPTARG}")
       carcinogens+=(${OPTARG});;
    a) options+=("-a ${OPTARG}");;
    b) options+=("-b ${OPTARG}");;
    x) options+=("-x ${OPTARG}");;
    h) options+=("-h ${OPTARG}");;
    o) options+=("-o ${OPTARG}");;
    m) options+=("-m ${OPTARG}");;
    r) options+=("-r ${OPTARG}");;
    k) options+=("-k");;
    j) options+=("-j ${OPTARG}");;
    p) options+=("-p");;
    q) options+=("-q");;
    u) options+=("-u ${OPTARG}");;
    e) options+=("-e ${OPTARG}");;
  esac
done

if [ $nCarcin -eq 1 ]; then
  if [ ${#carcinogens[@]} -eq 0 ]; then
    carcinogens+=(0)
  fi
fi

if [ $verbose -eq 1 ]; then
  echo 'Simulation Runner'
  echo 'Running' $numSim 'simulations.'
  echo ''
fi

options+=("-f $outFolder")
if [ ! -d $outFolder ]; then
  mkdir $outFolder
fi
cd $outFolder

if [ $verbose -eq 1 ]; then
  $currDir/main "${options[@]}" >> >(tee out.txt) 2> out.log
  retVal=$?
else
  $currDir/main "${options[@]}" > out.txt 2> out.log
  retVal=$?
fi

if [ ! -s out.log ]; then
  rm out.log
fi
sed -i '/progress/d' out.txt

pattern=('^Starting simulation ' '^Done simulation ')
for ((i=1; i <= numSim; i++)); do
  if [ ! -d $i ]; then
    break
  else
    cd $i
  fi

  sed -n '/'"${pattern[0]}"$i'/,/'"${pattern[1]}"$i'/p;/'"${pattern[1]}"$i'/q'\
    ../out.txt > $i.txt
  sed -i '/'"${pattern[0]}"$i'/,/'"${pattern[1]}"$i'/d;/'"${pattern[1]}"$i'/q'\
    ../out.txt

  mkdir time_steps
  mv 0*.jpeg time_steps
  if [ ! $nCarcin -eq 0 ]; then
    mkdir carcin_function
    for ((j=0; j < nCarcin; j++)); do
      mkdir carcin_function/carcin${carcinogens[j]}
      mv carcin${carcinogens[j]}*.jpeg carcin_function/carcin${carcinogens[j]}
    done
  fi
  mkdir max_gene_map
  mv genes_0*.jpeg max_gene_map
  mkdir lineage_heatmap
  mv heatmap_0*.jpeg lineage_heatmap
  mkdir top_lineage_map
  for ((j=0; j < 7; j++)); do
    mkdir top_lineage_map/$j
    mv max$((j))_0*.jpeg top_lineage_map/$j
  done
  mkdir videos
  mv *.mp4 videos
  mkdir data
  if [ -f *.data.lz4 ]; then
    mkdir data/grid_data
    mv *.data.lz4 data/grid_data
  fi
  gnuplot -c $currDir/bin/create_plots $(($gridSize * $gridSize)) $nCarcin ${carcinogens[@]} 2> ../plots.log
  mkdir data/data_files
  mv *.data data/data_files
  mkdir graphs
  mv *.png graphs

  cd ..
done
mv out.txt params.txt

if [ $verbose -eq 1 ]; then
  echo ''
fi

cd $currDir
if [ $verbose -eq 1 ]; then
  bin/get_stats.sh $outFolder > >(tee $outFolder/stats.txt)\
    2> $outFolder/stats.log
else
  bin/get_stats.sh $outFolder > $outFolder/stats.txt 2> $outFolder/stats.log
fi

if [ ! -s $outFolder/stats.log ]; then
  rm $outFolder/stats.log
fi

if [ $i -lt $numSim ] || [ $retVal -eq 1 ]; then
  exit 1
fi

exit 0