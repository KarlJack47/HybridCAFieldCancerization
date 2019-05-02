#!/bin/bash

verbose=0
num_sim=100
display=0
save=0
max_time=200
grid_size=512
maxt_tc=-1
while getopts vdsn:t:g:e: option; do
    case "${option}"
    in
        v) verbose=1;;
        d) display=1;;
        s) save=1;;
	n) num_sim=${OPTARG};;
        t) max_time=${OPTARG};;
        g) grid_size=${OPTARG};;
	e) maxt_tc=${OPTARG};;
    esac
done

if [ $verbose -eq 1 ]; then
    echo "Simulation Runner"
    if [ $maxt_tc -eq -1 ]; then
        echo "Running $num_sim simulations with a grid size of $grid_size and $max_time time steps."
    else
        echo "Running $num_sim simulations with a grid size of $grid_size, $max_time time steps, and $maxt_tc time steps before excisions."
    fi
    echo ""
fi

if [ $save -eq 1 ]; then
    cd output
fi

out_file=output_$(date -d "today" +"%Y%m%d%H%M%S").txt
for ((i=1; i <= num_sim; i++)); do
    if [ $verbose -eq 1 ]; then
        echo "Running simulation $i"
    fi

    if [ $save -eq 1 ]; then
        mkdir $i
        cd $i
        touch $i.txt
        if [ $verbose -eq 1 ]; then
            ../../main $display $save $max_time $grid_size $maxt_tc | tee $i.txt
        else
            ../../main $display $save $max_time $grid_size $maxt_tc > $i.txt
        fi
        cd ..
    else
        echo $i >> $out_file
        if [ $verbose -eq 1 ]; then
            ./main $display $save $max_time $grid_size $maxt_tc | tee -a $out_file
        else
            ./main $display $save $max_time $grid_size $maxt_tc >> $out_file
        fi
    fi

    if [ $verbose -eq 1 ]; then
        echo "Done simulation $i"
    fi
done

if [ $verbose -eq 1 ]; then
    echo ""
fi
if [ $save -eq 1 ]; then
    cd ..
    bin/get_stats.sh > output/stats.txt
else
    bin/get_stats.sh $out_file > stats_$(date -d "today" +"%Y%m%d%H%M%S").txt
fi

rm output_*.txt

exit 0
