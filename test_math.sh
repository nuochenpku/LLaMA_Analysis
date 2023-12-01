
MODEL_DIR=$1
mkdir -p $MODEL_DIR/gsm8k_mc_reason/


CUDA_VISIBLE_DEVICES=1 python gsm8k_mc_eval.py --model-name $MODEL_DIR --data-path  data/gsm8k_reason_mc_pure.json --output-path $MODEL_DIR/gsm8k_mc_reason/ --num-gpus 1 --layer_wise \
&> $MODEL_DIR/gsm8k_mc_reason/gsm8k_reason_mc_layers_ind.log &

