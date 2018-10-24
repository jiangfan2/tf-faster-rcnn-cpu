#!/bin/bash

cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..

NET=res101
TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
mkdir -p output/${NET}/${TRAIN_IMDB}/default