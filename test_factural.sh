
MODEL_DIR=$1

mkdir -p $MODEL_DIR/factural

CUDA_VISIBLE_DEVICES=0 python factural_eval.py --model-name $MODEL_DIR --data-path  data/mlama_short.csv --output-path $MODEL_DIR/factural/ --num-gpus 1 --layer_wise \
&> $MODEL_DIR/factural/layers.log &
