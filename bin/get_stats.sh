#!/bin/bash

if [ ! $# -eq 1 ]; then
    exit 0
fi

input=$1/*/*.txt
totalSim=$(grep "The CA will run for" $input | uniq | wc -l)
delim='/'
delim1='('
posSimNum=2
posVal=3
posVal1=1
exprStr=('first CSC' 'first TC' 'TC recurred')
cellType=('CSC' 'TC' 'TC')
formStr=('form' 'form' 'reform')
totalStr=('simulations' 'simulations' 'excisions')

for i in {0..2}; do
    if [ $i -eq 2 ]; then
        delim1=' '
        posVal1=1-6
    fi

    totalVals=$(grep "${exprStr[$i]}" $input | cut -f $posVal1 -d "$delim1" | wc -l)

    if [ $totalVals -eq 0 ]; then
        if [ ! $i -eq 2 ]; then
            echo 'No' ${cellType[$i]} 'were formed during the simulations.'
        else
            echo 'No TC reformed due to likely no excisions.'
        fi
        continue;
    fi

    simInfo=($(grep "${exprStr[$i]}" $input | cut -f $posVal1 -d "$delim1" |\
             sort -n -t "$delim" -k $posSimNum |\
             sed -e 's/[^0-9/ ]//g' | cut -d "$delim" -f $posVal))
    if [ $totalSim -eq 1 ]; then
        simInfo=(1 ${simInfo[@]})
        for j in {1..1000}; do
            if [ $j -eq $totalVals ]; then
                break;
            fi
            #limit=$((2*$j+$j))
            limit=$((2*$j))
            simInfo=(${simInfo[@]:0:$limit} 1 ${simInfo[@]:$limit})
        done
    fi

    simNum=${simInfo[0]}
    count=0
    total=0
    total1=0
    min=${simInfo[1]}
    min1=${simInfo[1]}
    max=${simInfo[1]}
    max1=${simInfo[1]}
    for idx in ${!simInfo[@]}; do
        if [ $(($idx % 2)) -eq 0 ]; then
            if [ ! $i -eq 2 ]; then
                simNum=${simInfo[$idx]}
                continue;
            fi
            if [ ! $simNum -eq ${simInfo[$idx]} ]; then
                avg=$(bc -l <<< "scale=6; $total1/$count")
                echo 'On average it took' $avg 'time steps for a TC to'\
                     'reform for simulation' $simNum 'after' $count 'excisions.'
                echo 'The minimum number of time steps for a TC to reform for'\
                     'simulation' $simNum 'after' $count 'excisions was' $min'.'
                echo 'The maximum number of time steps for a TC to reform for'\
                     'simulation' $simNum 'after' $count 'excisions was' $max'.'

                simNum=${simInfo[$idx]}
                count=0
                total1=0
                min1=${simInfo[$(($idx+1))]}
                max1=${simInfo[$(($idx+1))]}
            fi
            continue;
        fi

        if [ ! $i -eq 2 ]; then
            echo 'The first' ${cellType[$i]} 'formed at time step'\
                 ${simInfo[$idx]} 'for simulation' $simNum'.'
        fi

        total=$(($total+${simInfo[$idx]}))
        total1=$(($total1+${simInfo[$idx]}))
        if [ ${simInfo[$idx]} -lt $min ]; then
            min=${simInfo[$idx]}
        fi
        if [ ${simInfo[$idx]} -lt $min1 ]; then
            min1=${simInfo[$idx]}
        fi
        if [ ${simInfo[$idx]} -gt $max ]; then
            max=${simInfo[$idx]}
        fi
        if [ ${simInfo[$idx]} -gt $max1 ]; then
            max1=${simInfo[$idx]}
        fi
        if [ $i -eq 2 ]; then
            count=$(($count+1))
        fi

        if [[ $i -eq 2 && $idx -eq $((2*totalVals-1)) ]]; then
            avg=$(bc -l <<< "scale=6; $total1/$count")
            echo 'On average it took' $avg 'time steps for a TC to'\
                 'reform for simulation' $simNum 'after' $count 'excisions.'
            echo 'The minimum number of time steps for a TC to reform for'\
                 'simulation' $simNum 'after' $count 'excisions was' $min'.'
            echo 'The maximum number of time steps for a TC to reform for'\
                 'simulation' $simNum 'after' $count 'excisions was' $max'.'
       fi
    done
    avg=$(bc -l <<< "scale=6; $total/$totalVals")

    echo 'On average it took' $avg 'time steps for the first' ${cellType[$i]}\
         'to' ${formStr[$i]} 'for the' $totalVals ${totalStr[$i]}'.'
    echo 'The minimum number of time steps for the first' ${cellType[$i]}\
         'to' ${formStr[$i]} 'for the' $totalVals ${totalStr[$i]} 'was' $min'.'
    echo 'The maximum number of time steps for the first' ${cellType[$i]}\
         'to' ${formStr[$i]} 'for the' $totalVals ${totalStr[$i]} 'was' $max'.'
    echo ''
done

exit 0