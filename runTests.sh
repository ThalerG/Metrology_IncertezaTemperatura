#!/bin/bash

echo "Run 1/4"
python \fast_montecarlo.py --N_montecarlo 1000000 --fname difTamb_allPoints --Npoints 19 --dt 2 --t1 4 --Tamb_drift 2

echo "Run 2/4"
python \fast_montecarlo.py --N_montecarlo 1000000 --fname difTamb_widePoints --Npoints 3 --dt 10 --t1 10 --Tamb_drift 2

echo "Run 3/4"
python \fast_montecarlo.py --N_montecarlo 1000000 --fname difTamb_earlyPoints --Npoints 3 --dt 2 --t1 4 --Tamb_drift 2

echo "Run 4/4"
python \fast_montecarlo.py --N_montecarlo 1000000 --fname difTamb_latePoints --Npoints 3 --dt 2 --t1 20 --Tamb_drift 2