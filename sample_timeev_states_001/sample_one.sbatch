#!/bin/bash

date

if [ $# -ne 4 ]
then
	printf "Expecting four arguments\n" >&2;
	exit 1;
fi

echo "beginning sampling"
python sample_one.py $1 $2 $3 $4
echo "ending sampling"
date
