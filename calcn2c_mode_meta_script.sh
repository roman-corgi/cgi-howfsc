#!/bin/bash

# bash script to test calcn2c accuracy for each mode
# and maybe some different values of MKL_NUM_THREADS
# usage:
# $ ./calcn2c_mode_meta_script.sh 3.7
# $ ./calcn2c_mode_meta_script.sh 3.8

choices=(widefov narrowfov spectroscopy nfov_dm nfov_flat)

HOSTSHORT=${HOSTNAME%%.*}
mkl_num_threads=(1 4 8)
#nprocesses=(8 16 0) # 0 => number of cores
TIMEFORMAT='%1R, %1U'

echo mode, num_process, num_threads, time_seconds, real, user
for mode in ${choices[@]}; do

   logfn="log_${HOSTSHORT}_${mode}.txt"
   if test -f "$logfn"; then
      rm $logfn
   fi

   # loop through list of num_threads
   for num_threads in ${mkl_num_threads[@]}; do

       export MKL_NUM_THREADS=$num_threads
       start=$SECONDS

       # --num_process 0 is for the calcjac not for calcn2c
       time_result=$(time (python$1 precomptest.py --mode $mode --num_process 0 --do_calcn2c --logfile $logfn) 2>&1)
       duration=($(( SECONDS - start )))

       str="${mode}, ${pp}, ${num_threads}, ${duration}, ${time_result}, ${logfn}"
       echo $str

   done # num_threads
done # outer loop, for mode
