#!/bin/bash

# script to test use of environment variables to control multiprocessing and threads
# mode choices=(widefov narrowfov spectroscopy nfov_dm nfov_flat)
# usage:
# $ ./precomptest_env_meta_script.sh 3.7
# $ ./precomptest_env_meta_script.sh 3.8

HOSTSHORT=${HOSTNAME%%.*}
mkl_num_threads=('1' '4' '8') # env variables are strings
nprocesses=('8' '16' '0') # 0 => number of cores
TIMEFORMAT='%1R, %1U'

# force remove env variables to start from scratch
unset HOWFS_CALCJAC_NUM_PROCESS
unset HOWFS_CALCJAC_NUM_THREADS

# stdout will be a csv table with headers:
echo mode, num_process, num_threads, time_seconds, real, user

mode=widefov

logfn="log_envs_${HOSTSHORT}_${mode}.txt"
if test -f "$logfn"; then
   rm $logfn
fi

# first run with nothing defined, default = "safe and slow"
start=$SECONDS
time_result=$(time (python$1 precomptest.py --mode $mode --logfile $logfn) 2>&1)
duration=($(( SECONDS - start )))
# stdout will be a csv table
str="${mode}, None, None, ${duration}, ${time_result}, ${logfn}"
echo $str

# loop through list of num_threads and list of nprocesses
for num_threads in ${mkl_num_threads[@]}; do

    export HOWFS_CALCJAC_NUM_THREADS=$num_threads

    for pp in ${nprocesses[@]}; do

        export HOWFS_CALCJAC_NUM_PROCESS=$pp
           
        start=$SECONDS
        time_result=$(time (python$1 precomptest.py --mode $mode --logfile $logfn) 2>&1)
        duration=($(( SECONDS - start )))

        # stdout will be a csv table
        str="${mode}, ${pp}, ${num_threads}, ${duration}, ${time_result}, ${logfn}"
        echo $str
    done # for nprocesses
done # num_threads
