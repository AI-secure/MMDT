# MMDT Fairness Evaluation

## Text-to-image models
The scripts are summarized in ``run_t2i.sh``.

### Group/Individual fairness evaluation

#### Generation

```angular2html
for model in stabilityai/stable-diffusion-xl-base-1.0 dreamlike-art/dreamlike-photoreal-2.0 prompthero/openjourney-v4 DeepFloyd/IF-I-M-v1.0 dall-e-2 dall-e-3
do
  for dataset in occupation occupation_with_sex occupation_with_race occupation_with_age education education_with_sex education_with_race education_with_sex activity activity_with_sex
  do
    python main.py --inference_model $model --inference_gpu_id 0 --dataset $dataset --do_generation --save_dir  ./mmdt_results/fairness
  done
done
```

#### Score computation

```angular2html
for model in stabilityai/stable-diffusion-xl-base-1.0 dreamlike-art/dreamlike-photoreal-2.0 prompthero/openjourney-v4 DeepFloyd/IF-I-M-v1.0 dall-e-2 dall-e-3
do
  for dataset in occupation occupation_with_sex occupation_with_race occupation_with_age education education_with_sex education_with_race education_with_sex activity activity_with_sex
  do
    python main.py --inference_model $model --evaluate_gpu_id 0 --dataset $dataset --do_evaluate --save_dir ./mmdt_results/fairness
  done
done

target1=occupation
target2=education
target3=activity
for model in stabilityai/stable-diffusion-xl-base-1.0 dreamlike-art/dreamlike-photoreal-2.0 prompthero/openjourney-v4 DeepFloyd/IF-I-M-v1.0 dall-e-2 dall-e-3
do
  for attr in gender race age
  do
    python main.py --inference_model $model --evaluate_gpu_id 5 --do_fairness_score_calculation --score_comp_sensitive $attr --score_comp_target $target1 --save_dir  /data1/common/mintong/mmdt_results
  done
  for attr in gender race
  do
    python main.py --inference_model $model --evaluate_gpu_id 5 --do_fairness_score_calculation --score_comp_sensitive $attr --score_comp_target $target2 --save_dir  /data1/common/mintong/mmdt_results
  done
  for attr in gender
  do
    python main.py --inference_model $model --evaluate_gpu_id 5 --do_fairness_score_calculation --score_comp_sensitive $attr --score_comp_target $target3 --save_dir  /data1/common/mintong/mmdt_results
  done
done
```

## Image-to-text models
The scripts are summarized in ``run_i2t.sh``.

### Group fairness evaluation

#### Generation

```angular2html
for model in llava-hf/llava-v1.6-vicuna-7b-hf gpt-4-vision-preview gpt-4o-2024-05-13
do
  python main_i2t.py --inference_model $model --dataset occupation --do_generation
  python main_i2t.py --inference_model $model --dataset education --do_generation
  python main_i2t.py --inference_model $model --dataset activity --do_generation
done
```

#### Score computation

```angular2html
target1=occupation
target2=education
target3=activity

for model in llava-hf/llava-v1.6-vicuna-7b-hf gpt-4-vision-preview gpt-4o-2024-05-13
do
  for attr in gender race age
  do
    python main_i2t.py --inference_model $model --dataset $target1 --do_evaluate_group_fairness --sens_attr $attr
  done
  for attr in gender race
  do
    python main_i2t.py --inference_model $model --dataset $target2 --do_evaluate_group_fairness --sens_attr $attr
  done
  for attr in gender
  do
    python main_i2t.py --inference_model $model --dataset $target3 --do_evaluate_group_fairness --sens_attr $attr
  done
done
```

### Individual fairness evaluation

#### Generation
```angular2html
for model in llava-hf/llava-v1.6-vicuna-7b-hf gpt-4-vision-preview gpt-4o-2024-05-13
do
  python main_i2t.py --inference_model $model --dataset person_identification --do_generation --num_response_per_instance 1
done
```

#### Score computation
```angular2html
for model in llava-hf/llava-v1.6-vicuna-7b-hf gpt-4-vision-preview gpt-4o-2024-05-13
do
  for attr in gender race age
  do
    python main_i2t.py --inference_model $model --dataset person_identification --do_evaluate_individual_fairness --sens_attr $attr --num_response_per_instance 1
  done
done
```