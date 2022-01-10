#!/bin/bash

!pip install -q imageio
!pip install -q opencv-python
!pip install -q git+https://github.com/tensorflow/docs

python test_2019MT60739_2019MT60493.sh $1 $2 $3	
$ chmod +x test_2019MT60739_2019MT60493.sh
$ ./test_2019MT60739_2019MT60493.sh