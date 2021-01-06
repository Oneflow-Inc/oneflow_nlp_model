# pretrained model dir
PRETRAINED_MODEL=$1

# ofrecord dataset dir
DATA_ROOT=datasets/glue_ofrecord

# choose dateset `CoLA` or `MRPC`
#dataset=CoLA
dataset=MRPC
if [ $dataset = "CoLA" ]; then
  train_example_num=8551
  eval_example_num=1043
  test_example_num=1063
  learning_rate=1e-5
  wd=0.01
elif [ $dataset = "MRPC" ]; then
  train_example_num=3668
  eval_example_num=408
  test_example_num=1725
  learning_rate=2e-6
  wd=0.001
else
  echo "dataset must be 'CoLA' or 'MRPC'"
  exit
fi

train_data_dir=$DATA_ROOT/${dataset}/train
eval_data_dir=$DATA_ROOT/${dataset}/eval

python3 RoBERTa/run_classifier.py \
  --model=Glue_$dataset \
  --task_name=$dataset  \
  --gpu_num_per_node=1 \
  --num_epochs=4 \
  --train_data_dir=$train_data_dir \
  --train_example_num=$train_example_num \
  --eval_data_dir=$eval_data_dir \
  --eval_example_num=$eval_example_num \
  --model_load_dir=${PRETRAINED_MODEL} \
  --batch_size_per_device=32 \
  --eval_batch_size_per_device=4 \
  --loss_print_every_n_iter 20 \
  --log_dir=./log \
  --model_save_dir=./snapshots \
  --save_last_snapshot=True \
  --seq_length=128 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.1 \
  --hidden_dropout_prob=0.1 \
  --hidden_size_per_head=64 \
  --learning_rate $learning_rate \
  --weight_decay_rate $wd
