#!/bin/bash

for (( x=0 ; x<=80000 ; x++ )) ; do
  echo "printf(\"%d, %d, %d\n\", i, j, k);" >> chunk.txt
done
