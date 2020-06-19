#!/bin/bash

if [ ! -f main ]; then
    echo 'The program is not within the current directory.'
    exit 1
fi

help=0
verbose=1
numSim=100
display=0
save=1
maxTime=8766
gridSize=256
initType=0
perfectExcision=0
maxTTC=-1
while getopts hvdsn:t:g:i:pe: option; do
    case "${option}" in
        h) help=1;;
        v) verbose=1;;
        d) display=1;;
        s) save=1;;
	    n) numSim=${OPTARG};;
        t) maxTime=${OPTARG};;
        g) gridSize=${OPTARG};;
        i) initType=${OPTARG};;
        p) perfectExcision=1;;
	    e) maxTTC=${OPTARG};;
    esac
done

if [ $help -eq 1 ]; then
    echo 'Usage:' $0 '[OPTION]'
    echo 'Run CA a certain # of times with a set grid size and # of time steps.'
    echo ''
    echo 'Options:'
    echo '  -h     prints out the information you are currently reading'
    echo '  -v     flag used to display progress of the script'
    echo '  -d     enables the gui, default=disabled'
    echo '  -s     enables saving of pictures and videos, default=enabled'
    echo '  -n int number of simulations to run, default=100'
    echo '  -t int number of time steps, default=8766'
    echo '  -g int grid size, default=256 (power of 2 between 16 and 1024)'
    echo '  -i int init type, default=0, 0=one carcinogen,'
    echo '         1=start with tumour cells, 2=two carcinogen, 3=no carcinogen'
    echo '  -p     enables perfect tumour excision, default=disabled'
    echo '  -e int frequency of tumour excision, default=-1 meaning no excision'
    echo '         is performed unless excision mode is activated in the gui'
    exit 0
fi

if [ $verbose -eq 1 ]; then
    echo 'Simulation Runner'
    if [ $maxTTC -eq -1 ]; then
        echo 'Running' $numSim 'simulations with a grid size of' $gridSize\
             'and' $maxTime 'time steps.'
    else
        echo 'Running' $numSim 'simulations with a grid size of' $gridSize','\
             $maxTime 'time steps, and' $maxTTC 'time steps before excisions.'
    fi
    echo ''
fi

outFolder=output_$(date -d "today" +"%Y%m%d%H%M%S")
mkdir $outFolder
cd $outFolder

for ((i=1; i <= numSim; i++)); do
    j=$i
    if [ $verbose -eq 1 ]; then
        echo 'Running simulation' $i
    fi

    mkdir $i
    cd $i

    if [ $verbose -eq 1 ]; then
        ../../main $display $save $maxTime $gridSize $initType\
          $perfectExcision $maxTTC >> >(tee $i.txt) 2> $i.log
    else
        ../../main $display $save $maxTime $gridSize $initType\
          $perfectExcision $maxTTC > $i.txt 2> $i.log
    fi
    if [ ! -s $i.log ]; then
        rm $i.log
    else
        i=$(($numSim+1))
    fi
    sed -i '/progress/d' $i.txt
    gnuplot ../../bin/create_plots
    cd ..

    if [ $verbose -eq 1 ]; then
        echo 'Done simulation' $j
    fi
done

if [ $verbose -eq 1 ]; then
    echo ''
fi

cd ..
if [ $verbose -eq 1 ]; then
    bin/get_stats.sh $outFolder > >(tee $outFolder/stats.txt)\
    2> $outFolder/stats.log
else
    bin/get_stats.sh $outFolder > $outFolder/stats.txt 2> $outFolder/stats.log
fi

if [ ! -s $outFolder/stats.log ]; then
    rm $outFolder/stats.log
fi

exit 0