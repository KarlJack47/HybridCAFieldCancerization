#!/bin/bash

help=0
verbose=0
num_sim=100
display=0
save=0
max_time=600
grid_size=512
while getopts hvdsn:t:g: option; do
    case "${option}"
    in
        h) help=1;;
        v) verbose=1;;
        d) display=1;;
        s) save=1;;
        n) num_sim=${OPTARG};;
        t) max_time=${OPTARG};;
        g) grid_size=${OPTARG};;
    esac
done
if [ $help -eq 1 ]; then
    echo "Usage: $0 [OPTION]"
    echo "Run Hybrid CA a certain number of times with a certain grid size and number of time steps."
    echo ""
    echo "Options:"
    echo "  -h		prints out the information you are currently reading"
    echo "  -v		if flag is used then the progress of the script will be displayed"
    echo "  -d		whether to display the simulation as it is running"
    echo "  -s		whether to save the time steps into a file and create an output video"
    echo "  -n int	number of simulations to run, default is 100"
    echo "  -t int	number of time steps, default is 600 (note simulation hasn't been verified for >800 steps)"
    echo "  -g int	size of the CA grid, default is 512 (should be a power of 2 between 16 and 1024)"
    exit 0
fi

if [ $save -eq 1 ]; then
    if [ -d output ]; then
        mv output output_$(date -d "today" +"%Y%m%d%H%M%S")
        mkdir output
    fi
    if [ ! -d output ]; then
        mkdir output
    fi
fi

if [ $verbose -eq 1 ]; then
    if [ $save -eq 1 ]; then
        if [ $display -eq 1 ]; then
            bin/create_results.sh -v -d -s -n $num_sim -t $max_time -g $grid_size 2> /dev/null
        else
            bin/create_results.sh -v -s -n $num_sim -t $max_time -g $grid_size 2> /dev/null
        fi
    else
        if [ $display -eq 1 ]; then
            bin/create_results.sh -v -d -n $num_sim -t $max_time -g $grid_size 2> /dev/null
        else
            bin/create_results.sh -v -n $num_sim -t $max_time -g $grid_size 2> /dev/null
        fi
    fi
else
    if [ $save -eq 1 ]; then
        if [ $display -eq 1 ]; then
            bin/create_results.sh -d -s -n $num_sim -t $max_time -g $grid_size 2> /dev/null
        else
            bin/create_results.sh -s -n $num_sim -t $max_time -g $grid_size 2> /dev/null
        fi
    else
        if [ $display -eq 1 ]; then
            bin/create_results.sh -d -n $num_sim -t $max_time -g $grid_size 2> /dev/null
        else
            bin/create_results.sh -n $num_sim -t $max_time -g $grid_size 2> /dev/null
        fi
    fi
fi

exit 0
