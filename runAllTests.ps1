write-host "Run 1/7"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname beges1_allPoints --Npoints 19 --dt 2 --t1 4

write-host "Run 2/7"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname widePoints --Npoints 3 --dt 10 --t1 10

write-host "Run 3/7"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname narrowPoints --Npoints 3 --dt 2 --t1 10

write-host "Run 4/7"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname narrowPointsWorstCase --Npoints 3 --dt 2 --t1 36

write-host "Run 5/7"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname narrowPointsBestCase --Npoints 3 --dt 2 --t1 4

write-host "Run 6/7"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname beges2 --Npoints 11 --dt 2 --t1 20

write-host "Run 7/7"
python .\fast_montecarlo.py --N_montecarlo 1000000 --fname beges3 --Npoints 11 --dt 4 --t1 20