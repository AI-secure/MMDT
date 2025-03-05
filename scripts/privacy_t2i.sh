
#  bash scripts/privacy_t2i.sh stabilityai/stable-diffusion-2
modality="text_to_image"
model_id=$1  # e.g., stabilityai/stable-diffusion-2
perspective="privacy"
task="laion_1k" 

# enable dry_run to test environment
#python mmdt/main.py --modality ${modality} --model_id ${model_id} --perspectives ${perspective} --task $task --dry_run
# actual run 
python mmdt/main.py --modality ${modality} --model_id ${model_id} --perspectives ${perspective} --task $task 