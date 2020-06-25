#!/bin/bash

if [ ! -f main ]; then
    echo 'The program is not within the current directory.'
    exit 1
fi

help=0
verbose=0
numSim=10
options=()
while getopts hvdsn:t:g:i:c:a:b:x:pe: option; do
    case "${option}" in
        h) help=1;;
        v) verbose=1;;
        d) options+=("-d");;
        s) options+=("-s");;
	    n) options+=("-n ${OPTARG}")
           numSim=${OPTARG}
	       ;;
        t) options+=("-t ${OPTARG}");;
        g) options+=("-g ${OPTARG}");;
        i) options+=("-i ${OPTARG}");;
        c) options+=("-c ${OPTARG}");;
        a) options+=("-a ${OPTARG}");;
        b) options+=("-b ${OPTARG}");;
        x) options+=("-x ${OPTARG}");;
        p) options+=("-p");;
	    e) options+=("-e ${OPTARG}");;
    esac
done

if [ $help -eq 1 ]; then
    echo 'Usage:' $0 '[OPTION]'
    echo 'Run CA a certain # of times with a set grid size and # of time steps.'
    echo ''
    echo 'Options:'
    echo '  -h     prints out the information you are currently reading'
    echo '  -v     flag used to display progress of the script, default=disabled'
    echo '  -d     enables the gui, default=disabled'
    echo '  -s     enables saving of pictures and videos, default=disabled'
    echo '  -n int number of simulations to run, default=10'
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
    echo 'Running' $numSim 'simulations.'
    echo ''
fi

outFolder=output_$(date -d "today" +"%Y%m%d%H%M%S")
mkdir $outFolder
cd $outFolder
options+=("-f $outFolder")

if [ $verbose -eq 1 ]; then
    ../main "${options[@]}" >> >(tee out.txt) 2> out.log
else
    ../main "${options[@]}" > out.txt 2> out.log
fi

if [ ! -s out.log ]; then
    rm out.log
fi
sed -i '/progress/d' out.txt

for ((i=1; i <= numSim; i++)); do
    cd $i
    gnuplot ../../bin/create_plots
    sed 's/.*Starting simulation $i\(.*\)Done simulation $i/\1/' ../out.txt > $i.txt
    cd ..
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