#!/bin/bash

files=`find log/* | xargs ls -d | grep "log.txt"`

for file in $files;
do
    base_dir=`dirname $file`
    output_file=$base_dir/format.log
    echo $output_file
    echo "global_step,loss,accuracy,accuracy_per_sequence,accuracy_top5,approx_bleu_score,neg_log_perplexity,rouge_2_fscore,rouge_L_fscore" >$output_file
    grep "INFO:tensorflow:Saving dict for global step" $file | awk -F " " '{print $9$12$15$18$21$24$27$30$33}' >>$output_file
done
