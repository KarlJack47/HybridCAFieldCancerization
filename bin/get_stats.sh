#!/bin/bash

if [ ! $# -eq 1 ]; then
    exit 0
fi

input=$1/*/*.txt
delim='/'
posSimNum=2
posVal=3
exprStr=('first CSC' 'first TC' 'TC recurred')
cellType=('CSC' 'TC')
formStr=('form' 'form' 'reform')
totalStr=('simulations' 'simulations' 'excisions')

for i in {0..2}; do
    totalVals=$(grep "${exprStr[$i]}" $input | cut -f 1 -d '(' | uniq | wc -l)

    if [ $totalVals -eq 0 ]; then
        if [ ! $i -eq 2 ]; then
            echo 'No' ${cellType[$i]} 'were formed during the simulations.'
        else
            echo 'No TC reformed due to likely no excisions.'
        fi
        continue;
    fi

    simInfo=($(grep "${exprStr[$i]}" $input | cut -f 1 -d '(' | uniq |\
             sort -n -t "$delim" -k $posSimNum |\
             sed -e 's/[^0-9/ ]//g' | cut -d "$delim" -f $posVal))

    if [ $totalVals -eq 1 ]; then
        echo 'The first' ${cellType[$i]} 'formed at time step'\
             ${simInfo[0]} 'for the simulation.'
        continue;
    fi
    simNum=${simInfo[0]}
    count=0
    total=0
    min=${simInfo[1]}
    max=${simInfo[1]}
    for idx in ${!simInfo[@]}; do
        if [ $(($idx % 2)) -eq 0 ]; then
            if [ ! $i -eq 2 ]; then
                simNum=${simInfo[$idx]}
                continue;
            fi
            if [ ! $simNum -eq ${simInfo[$idx]} ]; then
                avg=$total/$count
                echo 'On average it took' $avg 'time steps for a TC to'\
                     'reform for simulation' $simNum 'after' $count 'excisions.'
                echo 'The minimum number of time steps for a TC to reform for'\
                     'simulation' $simNum 'after' $count 'excisions was' $min'.'
                echo 'The maximum number of time steps for a TC to reform for'\
                     'simulation' $simNum 'after' $count 'excisions was' $max'.'

                simNum=${simInfo[$idx]}
                count=0
                total=0
                min=${simInfo[$(($idx+1))]}
                max=${simInfo[$(($idx+1))]}
            fi
            continue;
        fi

        if [ ! $i -eq 2 ]; then
            echo 'The first' ${cellType[$i]} 'formed at time step'\
                 ${simInfo[$idx]} 'for simulation' $simNum'.'
        fi

        total=$(($total+${simInfo[$idx]}))
        if [ ${simInfo[$idx]} -lt $min ]; then
            min=${simInfo[$idx]}
        fi
        if [ ${simInfo[$idx]} -gt $max ]; then
            max=${simInfo[$idx]}
        fi
        if [ $i -eq 2 ]; then
            count=$(($count+1))
        fi
    done
    avg=$(($total/$totalVals))

    echo 'On average it took' $avg 'time steps for the first' ${cellType[$i]}\
         'to' ${formStr[$i]} 'for the' $totalVals ${totalStr[$i]}'.'
    echo 'The minimum number of time steps for the first' ${cellType[$i]}\
         'to' ${formStr[$i]} 'for the' $totalVals ${totalStr[$i]} 'was' $min'.'
    echo 'The maximum number of time steps for the first' ${cellType[$i]}\
         'to' ${formStr[$i]} 'for the' $totalVals ${totalStr[$i]} 'was' $max'.'
    echo ''
done

exit 0