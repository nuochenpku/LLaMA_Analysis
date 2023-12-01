
LAYERS=80
MODEL_DIR=1

mkdir -p $MODEL_DIR/logical/

CUDA_VISIBLE_DEVICES=4 python logical_eval.py --model-name $MODEL_DIR --data-path  data/reclor_val.json --output-path $MODEL_DIR/logical/ --num-gpus 1 --layer_wise \
&> $MODEL_DIR/logical/layer9s.log 