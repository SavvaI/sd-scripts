set -e
EXPERIMENT_NAME=$1
LEARNING_RATE=$2
STOP_ENCODER=$3

echo Experiment name: $EXPERIMENT_NAME
echo Learning rate: $LEARNING_RATE -  Recommended learning rate 1e-5
echo Number of iterations to keep text encoder unfreezed in the beginning: $STOP_ENCODER -  Recommended number -1

source /opt/anaconda3/bin/activate kohya-ss
python train_db.py \
    --pretrained_model_name_or_path=./checkpoints/model.ckpt \
    --dataset_config=./conf/dreambooth_dnd.toml \
    --output_dir="./checkpoints/dreambooth_dnd${EXPERIMENT_NAME}" \
    --output_name=dnd \
    --save_model_as=ckpt \
    --max_train_steps=50000 \
    --sample_every_n_steps=300 \
    --learning_rate=$LEARNING_RATE \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="fp16" \
    --train_batch_size=100 \
    --cache_latents \
    --gradient_checkpointing \
    --save_precision='fp16' \
    --sample_prompts=./conf/sample_prompt_dnd.txt \
    --vae_batch_size=100 \
    --prior_loss_weight=0.0 \
    --stop_text_encoder_training=$STOP_ENCODER
