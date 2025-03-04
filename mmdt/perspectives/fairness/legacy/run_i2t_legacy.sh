# Run group fairness
for model in llava-hf/llava-v1.6-vicuna-7b-hf gpt-4-vision-preview gpt-4o-2024-05-13
do
  python main_i2t.py --inference_model $model --dataset occupation --do_generation
  python main_i2t.py --inference_model $model --dataset education --do_generation
  python main_i2t.py --inference_model $model --dataset activity --do_generation
done

# Evaluate group fairness
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


# Run individual fairness
for model in llava-hf/llava-v1.6-vicuna-7b-hf gpt-4-vision-preview gpt-4o-2024-05-13
do
  python main_i2t.py --inference_model $model --dataset person_identification --do_generation --num_response_per_instance 1
done

# Evaluate individual fairness
for model in llava-hf/llava-v1.6-vicuna-7b-hf gpt-4-vision-preview gpt-4o-2024-05-13
do
  for attr in gender race age
  do
    python main_i2t.py --inference_model $model --dataset person_identification --do_evaluate_individual_fairness --sens_attr $attr --num_response_per_instance 1
  done
done
