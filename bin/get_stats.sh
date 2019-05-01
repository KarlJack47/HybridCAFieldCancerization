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

input=$out_file
delim='/'
pos_sim_num=2
pos_val=8
if echo "$out_file" | grep -q "_"; then
    input=$out_file.t
    delim=' '
    pos_sim_num=1
    pos_val=9
fi

search_str=('CSC' 'TC was formed')
cell_type=('CSC' 'TC')
for i in {0..1}; do
    num_sim_form=$(grep "${search_str[$i]}" $input | uniq | sed -e 's/[^0-9 ]//g' | awk '{ count++ } END { print count }')

    if [ "$num_sim_form" == "" ]; then
        echo 'No' ${cell_type[$i]} 'were formed during the simulations.'
    else
        sim_nums=($(grep "${search_str[$i]}" $input | uniq | sed -e 's/[^0-9/ ]//g' | cut -d "$delim" -f $pos_sim_num))
        sim_vals=($(grep "${search_str[$i]}" $input | uniq | sed -e 's/[^0-9/ ]//g' | cut -d ' ' -f $pos_val))
        echo ${sim_nums[@]}
	echo ${sim_vals[@]}
        total=0
        for val in ${sim_vals[@]}; do
	    total=$(($total+$val))
        done
        avg=$(($total/$num_sim_form))
        IFS=$'\n'
        min=$(echo "${sim_vals[*]}" | sort -n | head -1)
        max=$(echo "${sim_vals[*]}" | sort -n | tail -1)

        for index in ${!sim_nums[@]}; do
            echo 'The first' ${cell_type[$i]} 'formed at time step' ${sim_vals[$index]} 'for simulation' ${sim_nums[$index]}'.'
        done
        echo 'On average it took' $avg 'time steps for the first' ${cell_type[$i]} 'to form for the' $num_sim_form 'simulations.'
        echo 'The minimum number of time steps for the first' ${cell_type[$i]} 'to form for the' $num_sim_form 'simulations was' $min'.'
        echo 'The maximum number of time steps for the first' ${cell_type[$i]} 'to form for the' $num_sim_form 'simulations was' $max'.'
        echo ''

	if [ "${cell_type[$i]}" == "TC" ]; then
	    total_vals=$(grep 'TC was reformed' $input | uniq | wc -l)
	    if [ "$total_vals" == "" ]; then
		continue
	    fi
	    sim_nums=($(grep 'TC was reformed' $input | uniq | sed -e 's/[^0-9/ ]//g' | cut -d "$delim" -f $pos_sim_num))
	    sim_vals=($(grep 'TC was reformed' $input | uniq | sed -e 's/[^0-9/ ]//g' | cut -d ' ' -f $(($pos_val+1))))
	    sim_num=${sim_nums[0]}
	    num_excise=()
	    count=0
	    for val in ${sim_nums[@]}; do
	        if [ "$sim_num" == "$val" ]; then
		    count=$(($count+1))
		else
		    num_excise+=($count)
		    count=1
		    sim_num=$val
		fi
	    done
            num_excise+=($count)
	    start=0
            for index in ${!num_excise[@]}; do
                total=0
		temp=()
		limit=$(($start+${num_excise[$index]}))
		sim_num=${sim_nums[$start]}
		for idx in ${!sim_vals[@]}; do
		    if [ "$idx" == "$limit" ]; then
			start=$idx
			break;
		    fi
		    if [[ $idx -ge $start ]]; then
		        temp+=(${sim_vals[$idx]})
		    fi
		done
		for val in ${temp[@]}; do
		    total=$(($total+$val))
		done
		avg=$(($total/${num_excise[$index]}))
		min=$(echo "${temp[*]}" | sort -n | head -1)
        	max=$(echo "${temp[*]}" | sort -n | tail -1)
		echo 'On average it took' $avg 'time steps for a TC to reform for simulation' $sim_num'.'
		echo 'The minimum number of time steps for the first TC to reform for simulation' $sim_num 'was' $min'.'
		echo 'The maximum number of time steps for the first TC to reform for simulation' $sim_num 'was' $max'.'
            done
            echo ''

	    total=0
	    for val in ${sim_vals[@]}; do
	        total=$(($total+$val))
	    done
            avg=$(($total/$total_vals))
	    min=$(echo "${sim_vals[*]}" | sort -n | head -1)
	    max=$(echo "${sim_vals[*]}" | sort -n | tail -1)
	    echo 'On average it took' $avg 'time steps for a TC to reform for the simulations.'
	    echo 'The minimum number of time steps for a TC to reform for the simulations was' $min'.'
	    echo 'The maximum number of time steps for a TC to reform for the simulations was' $max'.'
	    echo ''
	fi
    fi
done

if echo "$out_file" | grep -q "_"; then
    rm *.t
fi

exit 0
