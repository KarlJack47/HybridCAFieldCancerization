#!/bin/bash

echo "Simulation Runner"
echo ""

cd output

for i in {1..100}; do
	echo "Running simulation $i"
	mkdir $i
	cd $i
	touch $i.txt
	../../main > $i.txt
	cd ..
	echo "Done simulation $i"
done

exit 0
