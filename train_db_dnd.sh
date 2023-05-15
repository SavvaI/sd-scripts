set -e
source /opt/anaconda3/bin/activate kohya-ss
python train_db.py \
    --pretrained_model_name_or_path=./checkpoints/model.ckpt \
    --dataset_config=./conf/dreambooth_dnd.toml \
    --output_dir=./checkpoints/dreambooth_dnd \
    --output_name=dnd \
    --save_model_as=ckpt \
    --max_train_steps=50000 \
    --sample_every_n_steps=300 \
    --learning_rate=2e-5 \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="fp16" \
    --train_batch_size=125 \
    --cache_latents \
    --gradient_checkpointing \
    --save_precision='fp16' \
    --sample_prompts=./conf/sample_prompt_dnd.txt \
    --vae_batch_size=300 \
    --prior_loss_weight=0.0 \
    --stop_text_encoder_training=-1