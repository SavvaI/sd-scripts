set -e
source /opt/anaconda3/bin/activate kohya-ss
SCENE_NAME=$1
SCENE_PROMPT="$2"
PATH_TO_CKPT=./checkpoints/model.ckpt
PATH_TO_SK3D=./data/sk3d
DREAMBOOTH_NAME=dreambooth_data:${SCENE_NAME}:${SCENE_PROMPT// /_}
#PATH_TO_SK3D=/mnt/datasets/sk3d
#PATH_TO_CKPT=/root/project/startup/webui/models/Stable-diffusion/model.ckpt
#SCENE_WHITESPACE="${SCENE_NAME//_/ }"
echo "scene name: $SCENE_NAME"
echo "prompt: $SCENE_PROMPT"
echo "dreambooth folder name: $DREAMBOOTH_FOLDER"
echo "Stable Diffusion checkpoint path: $PATH_TO_CKPT"
echo "path to sk3d dataset: $PATH_TO_SK3D"
#echo "$SCENE_WHITESPACE"



cp -f ./conf/dreambooth_data_template.toml ./conf/dreambooth_data_${SCENE_NAME}.toml
sed -i "s/CASE_NAME/$SCENE_NAME/" ./conf/dreambooth_data_${SCENE_NAME}.toml
#sed -i "s/CASE_WHITESPACE/$SCENE_WHITESPACE/" ./conf/dreambooth_data_${SCENE_NAME}.toml
sed -i "s/CASE_PROMPT/$SCENE_PROMPT/" ./conf/dreambooth_data_${SCENE_NAME}.toml
sed -i "s|CASE_SK3D|$PATH_TO_SK3D|" ./conf/dreambooth_data_${SCENE_NAME}.toml


cp -f ./conf/sample_prompt_template.txt ./conf/sample_prompt_${SCENE_NAME}.txt
#sed -i "s/CASE_WHITESPACE/$SCENE_WHITESPACE/" ./conf/sample_prompt_${SCENE_NAME}.txt
sed -i "s/CASE_PROMPT/$SCENE_PROMPT/" ./conf/sample_prompt_${SCENE_NAME}.txt


echo "Generating regularisation images: "
echo "sample out dir: ./data/${SCENE_NAME}_reg_sample"
echo "sample prompt: ${SCENE_PROMPT}"

rm -rf ./data/${SCENE_NAME}_reg_sample
python gen_img_diffusers.py \
    --ckpt $PATH_TO_CKPT \
    --outdir ./data/${SCENE_NAME}_reg_sample \
    --xformers --fp16 --W 512 --H 512 --scale 12.5 \
    --steps 40 --batch_size 20 --images_per_prompt 500 \
    --prompt "${SCENE_PROMPT}"

rm -rf ./checkpoints/dreambooth_${SCENE_NAME}

python train_db.py \
    --pretrained_model_name_or_path=$PATH_TO_CKPT \
    --dataset_config=./conf/dreambooth_data_${SCENE_NAME}.toml \
    --output_dir=./checkpoints/dreambooth_${SCENE_NAME} \
    --output_name=1st_stage \
    --save_model_as=safetensors \
    --max_train_steps=500 \
    --sample_every_n_steps=250 \
    --learning_rate=1e-6 \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="fp16" \
    --train_batch_size=1 \
    --cache_latents \
    --gradient_checkpointing \
    --save_precision='fp16' \
    --sample_prompts=./conf/sample_prompt_${SCENE_NAME}.txt \
    --prior_loss_weight=1.0 \
    --stop_text_encoder_training=500
    
python convert_diffusers20_original_sd.py ./checkpoints/dreambooth_${SCENE_NAME}/1st_stage.safetensors \
    ./checkpoints/dreambooth_${SCENE_NAME}/diffusers_models/1st_stage  \
    --v1 --reference_model="CompVis/stable-diffusion-v1-4" --fp16

python train_db.py \
    --pretrained_model_name_or_path=./checkpoints/dreambooth_${SCENE_NAME}/1st_stage.safetensors \
    --dataset_config=./conf/dreambooth_data_${SCENE_NAME}.toml \
    --output_dir=./checkpoints/dreambooth_${SCENE_NAME} \
    --output_name=2nd_stage \
    --save_model_as=safetensors \
    --max_train_steps=500 \
    --sample_every_n_steps=250 \
    --learning_rate=1e-6 \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="fp16" \
    --train_batch_size=1 \
    --cache_latents \
    --gradient_checkpointing \
    --save_precision='fp16' \
    --sample_prompts=./conf/sample_prompt_${SCENE_NAME}.txt \
    --prior_loss_weight=1.0 \
    --stop_text_encoder_training=-1
    
python convert_diffusers20_original_sd.py ./checkpoints/dreambooth_${SCENE_NAME}/2nd_stage.safetensors \
    ./checkpoints/dreambooth_${SCENE_NAME}/diffusers_models/2nd_stage  \
    --v1 --reference_model="CompVis/stable-diffusion-v1-4" --fp16
    
printf "shs ${SCENE_PROMPT}" > ./checkpoints/dreambooth_${SCENE_NAME}/positive_prompt.txt
touch ./checkpoints/dreambooth_${SCENE_NAME}/negative_prompt.txt
    
    
