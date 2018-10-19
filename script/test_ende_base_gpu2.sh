#export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=4,7
PROBLEM=translate_ende_wmt32k 
#PROBLEM=translate_enzh_ac8k
MODEL=my_transformer
HPARAMS=transformer_base


HOME=/DATA/disk1/lixiaolong2/ai-challenger/en-de
USR_DIR=$HOME/usr_dir
DATA_DIR=$HOME/t2t_data
TMP_DIR=$HOME/raw_data
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM \
  --t2t_usr_dir=$USR_DIR

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --worker_gpu=2 \
  --t2t_usr_dir=$USR_DIR


TEST_FILE=$HOME/test
FROM_FILE=$TEST_FILE/en_de.inputs
DECODE_FILE=$TEST_FILE/en_de.decodes
TARGET_FILE=$TEST_FILE/en_de.targets

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_to_file=$DECODE_FILE \
  --t2t_usr_dir=$USR_DIR \
#  --decode_from_file=$FROM_FILE \



# See the translations
# cat $DECODE_2_FILE

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
t2t-bleu --translation=$DECODE_FILE --reference=$TARGET_FILE 
