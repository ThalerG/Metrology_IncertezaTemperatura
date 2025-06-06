#!/bin/bash

echo "Run 1/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname difTamb_allPoints --Npoints 19 --dt 2 --t1 4 --Tamb_2 26

echo "Run 2/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname difTamb_widePoints --Npoints 3 --dt 10 --t1 10 --Tamb_2 26

echo "Run 3/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname difTamb_earlyPoints --Npoints 3 --dt 2 --t1 4 --Tamb_2 26

echo "Run 4/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname difTamb_latePoints --Npoints 3 --dt 2 --t1 20 --Tamb_2 26

echo "Run 5/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname normC_allPoints --Npoints 19 --dt 2 --t1 4 --uniformC --s_c 0.3

echo "Run 6/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname normC_widePoints --Npoints 3 --dt 10 --t1 10 --uniformC --s_c 0.3

echo "Run 7/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname normC_earlyPoints --Npoints 3 --dt 2 --t1 4 --uniformC --s_c 0.3

echo "Run 8/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname normC_latePoints --Npoints 3 --dt 2 --t1 20 --uniformC --s_c 0.3

echo "Run 5/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname exponentialTf_allPoints --Npoints 19 --dt 2 --t1 4 --exponentialTf --s_t0 0.1

echo "Run 6/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname exponentialTf_widePoints --Npoints 3 --dt 10 --t1 10 --exponentialTf--s_t0 0.1

echo "Run 7/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname exponentialTf_earlyPoints --Npoints 3 --dt 2 --t1 4 --exponentialTf--s_t0 0.1

echo "Run 8/12"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname exponentialTf_latePoints --Npoints 3 --dt 2 --t1 20 --exponentialTf--s_t0 0.1