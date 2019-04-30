#!/bin/bash

out_file=output/*/*.txt
if [ ! $# -eq 0 ]; then
    out_file=$1
fi

if echo "$out_file" | grep -q "_"; then
    sim_num=1
    while IFS= read -r line; do
        if [[ "$line" == [0-9]* ]]; then
            sim_num=$line
        else
            echo "$sim_num" "$line"
        fi
    done < $out_file > $out_file.t
fi

if echo "$out_file" | grep -q "_"; then
    num_CSC="$(grep CSC $out_file.t | uniq | sed 's/^.*:/:/' | sed -e 's/[^0-9 ]//g' | awk '{ count++ } END { print count }')"
    num_TC="$(grep 'TC was formed at time step' $out_file.t | uniq | sed 's/^.*:/:/' | sed -e 's/[^0-9 ]//g' | awk '{ count++ } END { print count }')"
else
    num_CSC="$(grep CSC $out_file | uniq | sed 's/^.*:/:/' | sed -e 's/[^0-9 ]//g' | awk '{ count++ } END { print count }')"
    num_TC="$(grep 'TC was formed at time step' $out_file | uniq | sed 's/^.*:/:/' | sed -e 's/[^0-9 ]//g' | awk '{ count++ } END { print count }')"
fi

if [ "$num_CSC" == "" ]; then
    echo 'No CSC were formed during the simulations.'
fi
if [ "$num_TC" == "" ]; then
    echo 'No TC were formed during the simulations.'
fi

sim_nums_CSC=()
sim_vals_CSC=()
sim_nums_TC=()
sim_vals_TC=()
if [ ! "$num_CSC" == "" ]; then
    if echo "$out_file" | grep -q "_"; then
        sim_nums_s="$(grep CSC $out_file.t | uniq | grep -oP '.* A' | sed -e 's/[^0-9]//g')"
        sim_vals_s="$(grep CSC $out_file.t | uniq | sed 's/^. A*//' | sed -e 's/[^0-9]//g')"
        sim_nums_CSC=($sim_nums_s)
        sim_vals_CSC=($sim_vals_s)
        avg_CSC="$(grep CSC $out_file.t | uniq | sed 's/^. A*//' | sed -e 's/[^0-9 ]//g' | awk '{ total += $_; count++ } END { print total/count }')"
        min_CSC="$(grep CSC $out_file.t | uniq | sed 's/^. A*//' | sed -e 's/[^0-9 ]//g' | sort -n | head -1)"
        max_CSC="$(grep CSC $out_file.t | uniq | sed 's/^. A*//' | sed -e 's/[^0-9 ]//g' | sort -n | tail -1)"
    else
        sim_nums_s="$(grep CSC $out_file | sort -t / -k2n | uniq | grep -oP '.*(?=/)' | sed -e 's/[^0-9]//g')"
        sim_vals_s="$(grep CSC $out_file | sort -t / -k2n | uniq | sed 's/^.*:/:/' | sed -e 's/[^0-9]//g')"
        sim_nums_CSC=($sim_nums_s)
        sim_vals_CSC=($sim_vals_s)
        avg_CSC="$(grep CSC $out_file | uniq | sed 's/^.*://' | sed -e 's/[^0-9 ]//g' | awk '{ total += $_; count++ } END { print total/count }')"
        min_CSC="$(grep CSC $out_file | uniq | sed 's/^.*://' | sed -e 's/[^0-9 ]//g' | sort -n | head -1)"
        max_CSC="$(grep CSC $out_file | uniq | sed 's/^.*://' | sed -e 's/[^0-9 ]//g' | sort -n | tail -1)"
    fi
fi

if [ ! "$num_TC" == "" ]; then
    if echo "$out_file" | grep -q "_"; then
        sim_nums_s="$(grep 'TC was formed at time step' $out_file.t | uniq | grep -oP '.* A' | sed -e 's/[^0-9]//g')"
        sim_vals_s="$(grep 'TC was formed at time step' $out_file.t | uniq | sed 's/^. A*//' | sed -e 's/[^0-9]//g')"
        sim_nums_TC=($sim_nums_s)
        sim_vals_TC=($sim_vals_s)
        avg_TC="$(grep 'TC was formed at time step' $out_file.t | uniq | sed 's/^. A*//' | sed -e 's/[^0-9 ]//g' | awk '{ total += $_; count++ } END { print total/count }')"
        min_TC="$(grep 'TC was formed at time step' $out_file.t | uniq | sed 's/^. A*//' | sed -e 's/[^0-9 ]//g' | sort -n | head -1)"
        max_TC="$(grep 'TC was formed at time step' $out_file.t | uniq | sed 's/^. A*//' | sed -e 's/[^0-9 ]//g' | sort -n | tail -1)"
        rm *.t
    else
        sim_nums_s="$(grep 'TC was formed at time step' $out_file | sort -t / -k2n | uniq | grep -oP '.*(?=/)' | sed -e 's/[^0-9]//g')"
        sim_vals_s="$(grep 'TC was formed at time step' $out_file | sort -t / -k2n | uniq | sed 's/^.*://' | sed -e 's/[^0-9]//g')"
        sim_nums_TC=($sim_nums_s)
        sim_vals_TC=($sim_vals_s)
        avg_TC="$(grep 'TC was formed at time step' $out_file | uniq | sed 's/^.*://' | sed -e 's/[^0-9 ]//g' | awk '{ total += $_; count++ } END { print total/count }')"
        min_TC="$(grep 'TC was formed at time step' $out_file | uniq | sed 's/^.*://' | sed -e 's/[^0-9 ]//g' | sort -n | head -1)"
        max_TC="$(grep 'TC was formed at time step' $out_file | uniq | sed 's/^.*://' | sed -e 's/[^0-9 ]//g' | sort -n | tail -1)"
    fi
fi

if [ ! "$num_CSC" == "" ]; then
    for index in ${!sim_nums_CSC[@]}; do
        echo 'The first CSC formed at time step' ${sim_vals_CSC[$index]} 'for simulation' ${sim_nums_CSC[$index]}'.'
    done
    echo 'On average it took' $avg_CSC 'time steps for the first CSC to form for the' $num_CSC 'simulations.'
    echo 'The minimum number of time steps for the first CSC to form for the' $num_CSC 'simulations was' $min_CSC'.'
    echo 'The maximum number of time steps for the first CSC to form for the' $num_CSC 'simulations was' $max_CSC'.'
    echo ''
fi

if [ ! "$num_TC" == "" ]; then
    for index in ${!sim_nums_TC[@]}; do
        echo 'The first TC formed at time step' ${sim_vals_TC[$index]} 'for simulation' ${sim_nums_TC[$index]}'.'
    done
    echo 'On average it took' $avg_TC 'time steps for the first TC to form for the' $num_TC 'simulations.'
    echo 'The minimum number of time steps for the first TC to form for the' $num_TC 'simulations was' $min_TC'.'
    echo 'The maximum number of time steps for the first TC to form for the' $num_TC 'simulations was' $max_TC'.'
fi

exit 0
