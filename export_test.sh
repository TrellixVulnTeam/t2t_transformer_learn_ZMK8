
PROBLEM=translate_ende_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu

root_path=/DATA/disk1/guoxinze/tensor2tensor-master
DATA_DIR=${root_path}/t2t_data/ende
TRAIN_DIR=${root_path}/t2t_train/$PROBLEM/$MODEL-$HPARAMS
t2t-exporter \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --problem=$PROBLEM \
    --data_dir=$DATA_DIR \
    --output_dir=$TRAIN_DIR
