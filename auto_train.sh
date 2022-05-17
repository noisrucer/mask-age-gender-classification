#!/bin/bash

# python train.py -c config.json --m densenet161 --rid binary_60_1 --bs 32 --e 20 --fn 1 --imgS 224
# python train.py -c config.json --m densenet161 --rid binary_60_2 --bs 32 --e 13 --fn 2 --imgS 224
# python train.py -c config.json --m densenet161 --rid binary_60_3 --bs 32 --e 13 --fn 3 --imgS 224
python train.py -c config.json --m densenet161 --rid binary_60_4 --bs 32 --e 13 --fn 4 --imgS 224
python train.py -c config.json --m densenet161 --rid binary_60_5 --bs 32 --e 13 --fn 5 --imgS 224


