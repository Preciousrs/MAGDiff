

export MODEL_NAME="models/Realistic_Vision_V4.0_noVAE"  #Realistic_Vision_V4.0_noVAE
# CLIP Model

export ENCODER_NAME="models/image_encoder" # sd1.5 image coder

# # pretrained InstantID model
# --controlnet_model_name_or_path $CONTROLNET_NAME \


# Dataset
export ROOT_DATA_DIR="{data_root_path}" #data root

export JSON_FILE="{data_json}"   #data path
export MAIN_PROCESS_PORT=$((20000 + RANDOM % 9999))


export NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Output
export OUTPUT_DIR="alltrain/"  
#从头开始用weight=1

echo "OUTPUT_DIR: $OUTPUT_DIR"
#accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16" \
#CUDA_VISIBLE_DEVICES=0 \
export CUDA_VISIBLE_DEVICES="0"

python3 -m accelerate.commands.launch --main_process_port=$MAIN_PROCESS_PORT  train.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --image_encoder_path $ENCODER_NAME \
  --data_root_path $ROOT_DATA_DIR \
  --data_json_file $JSON_FILE \
  --output_dir $OUTPUT_DIR \
  --mixed_precision="fp16" \
  --resolution 256 \
  --learning_rate 1e-4 \
  --weight_decay=0.01 \
  --num_train_epochs 5 \
  --train_batch_size=9 \
  --dataloader_num_workers=1 \
  --save_epoch=1     

