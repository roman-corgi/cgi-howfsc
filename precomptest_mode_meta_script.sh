#!/bin/bash

# bash script to test jac computation times for various choices
# of number threads and number of parallel processes
#
# usage:
# $ ./precomptest_mode_meta_script.sh 3.7
# $ ./precomptest_mode_meta_script.sh 3.8
# 

# input argument 3.7 or 3.8
if [ "$#" -lt 1 ]; then
    pyver=3.7
else
    pyver=$1
fi

choices=(widefov narrowfov spectroscopy nfov_dm nfov_flat)

HOSTSHORT=${HOSTNAME%%.*}
mkl_num_threads=(1 4 8)
nprocesses=(8 16 0) # 0 => number of cores
TIMEFORMAT='%1R, %1U'

echo mode, num_process, num_threads, time_seconds, real, user
for mode in ${choices[@]}; do

   logfn="log_${HOSTSHORT}_${mode}.txt"
   if test -f "$logfn"; then
      rm $logfn
   fi

   # first run with 1 process and default mkl threads
   start=$SECONDS
   time_result=$(time (python$pyver precomptest.py --mode $mode --logfile $logfn) 2>&1)
   duration=($(( SECONDS - start )))

   str="${mode}, None, None, ${duration}, ${time_result}, ${logfn}"
   echo $str

   # loop through list of num_threads and list of nprocesses
   for num_threads in ${mkl_num_threads[@]}; do

       for pp in ${nprocesses[@]}; do

          start=$SECONDS
          time_result=$(time (python$1 precomptest.py --mode $mode --num_process $pp --num_threads $num_threads --logfile $logfn) 2>&1)
          duration=($(( SECONDS - start )))

          str="${mode}, ${pp}, ${num_threads}, ${duration}, ${time_result}, ${logfn}"
          echo $str
       done # for nprocesses
   done # num_threads
done # outer loop, for mode
