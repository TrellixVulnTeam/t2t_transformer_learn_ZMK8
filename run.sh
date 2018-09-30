#!/bin/bash
 
# usage:
#   ./run.sh script.sh
# for example:
#   ./run.sh scripts/test_run.sh
set -x
filename=$1
Time=Time_`date "+%Y-%m-%d-%H-%M-%S"`
sh_file=`basename $filename .sh`
base_dir=`dirname $filename`
base_dir=`echo ${base_dir#*/}`
mkdir -p log/$sh_file/${Time}
log_file=log/$sh_file/${Time}/log.txt
#nohup bash scripts/gather.sh >${log_file} 2>&1 &
#nohup bash $1 >${log_file} 2>&1 &
bash $1 >${log_file} 2>&1 &
