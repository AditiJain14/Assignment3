#!/bin/bash

!pip install -q imageio
!pip install -q opencv-python
!pip install -q git+https://github.com/tensorflow/docs

python train_2019MT60739_2019MT60493.py $1 $2
$ chmod +x train_2019MT60739_2019MT60493.sh
$ ./train_2019MT60739_2019MT60493.sh
