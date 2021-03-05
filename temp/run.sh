#!/bin/bash

RUN_NUMBER="001"

echo "Checking if directory for $1 already exists"

if [ ! -d output/$1 ]; then
	echo "Directory does not exist. Creating it now."
	mkdir output/$1
else
	echo "Checking run number"
	PREV_RUN_NUMBER=`exec ls output/$1 | sort -n | tail -1`
	RUN_NUMBER=`expr 1 + $PREV_RUN_NUMBER`
	RUN_NUMBER=`printf %03d $RUN_NUMBER`
fi

echo "Run number is $RUN_NUMBER"
mkdir output/$1/$RUN_NUMBER
