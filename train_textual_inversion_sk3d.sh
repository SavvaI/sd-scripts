set -e
source /opt/anaconda3/bin/activate kohya-ss
SCENE_NAME=$1
INIT_WORD="$2"

PATH_TO_CKPT=./checkpoints/model.ckpt
#PATH_TO_SK3D=./data/sk3d
PATH_TO_SK3D=/gpfs/gpfs0/savva.ignatyev/kaust/Diffuse-Neus/public_data/sk3d

#PATH_TO_SK3D=/mnt/datasets/sk3d
#PATH_TO_CKPT=/root/project/startup/webui/models/Stable-diffusion/model.ckpt

echo "scene name: $SCENE_NAME"
echo "init word: $INIT_WORD"

echo "Stable Diffusion checkpoint path: $PATH_TO_CKPT"
echo "path to sk3d dataset: $PATH_TO_SK3D"


cp -f ./conf/textual_inversion_data_template.toml ./conf/textual_inversion_data_${SCENE_NAME}.toml
sed -i "s/CASE_NAME/$SCENE_NAME/" ./conf/textual_inversion_data_${SCENE_NAME}.toml
sed -i "s|CASE_SK3D|$PATH_TO_SK3D|" ./conf/textual_inversion_data_${SCENE_NAME}.toml


rm -rf ./checkpoints/textual_inversion_${SCENE_NAME}


python train_textual_inversion.py \
    --pretrained_model_name_or_path=$PATH_TO_CKPT \
    --dataset_config=./conf/textual_inversion_data_${SCENE_NAME}.toml \
    --output_dir=./checkpoints/textual_inversion_${SCENE_NAME} \
    --output_name=textual_inversion \
    --prior_loss_weight=0.0 \
    --max_train_steps=1000 \
    --sample_every_n_steps=1000 \
    --sample_prompts=./conf/sample_prompt_textual_inversion.txt \
    --learning_rate=2e-3 \
    --optimizer_type="AdamW8bit" \
    --mixed_precision="fp16" \
    --train_batch_size=5 \
    --cache_latents \
    --save_precision='fp16' \
    --token_string=shs \
    --init_word="$INIT_WORD" \
    --num_vectors_per_token=4
    
# python convert_diffusers20_original_sd.py ./checkpoints/textual_inversion_${SCENE_NAME}/textual_inversion.safetensors \
#     ./checkpoints/textual_inversion_${SCENE_NAME}/diffusers_models/textual_inversion  \
#     --v1 --reference_model="CompVis/stable-diffusion-v1-4" --fp16
    
# python train_textual_inversion.py \
#     --pretrained_model_name_or_path=$PATH_TO_CKPT \
#     --dataset_config=./conf/textual_inversion_data_${SCENE_NAME}.toml \
#     --output_dir=./checkpoints/textual_inversion_${SCENE_NAME} \ 
#     --output_name=textual_inversion \
#     --save_model_as=safetensors \
#     --max_train_steps=1000 \
#     --learning_rate=1e-6 \
#     --optimizer_type="AdamW8bit" \
#     --xformers \
#     --mixed_precision="fp16" \
#     --cache_latents \
#     --gradient_checkpointing \
#     --save_precision='fp16' \
#     --token_string=shs \
#     --init_word=$INIT_WORD \
#     --num_vectors_per_token=4

# python train_db.py \
#     --pretrained_model_name_or_path=$PATH_TO_CKPT \
#     --dataset_config=./conf/textual_inversion_data_${SCENE_NAME}.toml \
#     --output_dir=./checkpoints/textual_inversion_${SCENE_NAME} \
#     --output_name=1st_stage \
#     --save_model_as=safetensors \
#     --max_train_steps=3000 \
#     --sample_every_n_steps=300 \
#     --learning_rate=1e-6 \
#     --optimizer_type="AdamW8bit" \
#     --xformers \
#     --mixed_precision="fp16" \
#     --train_batch_size=1 \
#     --cache_latents \
#     --gradient_checkpointing \
#     --save_precision='fp16' \
#     --sample_prompts=./conf/sample_prompt_${SCENE_NAME}.txt \
#     --prior_loss_weight=1.0 \
#     --stop_text_encoder_training=1000
