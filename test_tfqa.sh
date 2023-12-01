MODEL_DIR=$1
DATA_DIR=$2
mkdir -p $MODEL_DIR/tfqa_mc/

CUDA_VISIBLE_DEVICES=3 python tfqa_mc_eval_layer.py --model-name $MODEL_DIR --data-path  data --output-path $MODEL_DIR/tfqa_mc/ --num-gpus 1 --layer_wise 