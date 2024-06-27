python eval_image_to_text.py -m llava -i 0 > eval_image2text_llava_0.out
python eval_image_to_text.py -m llava -i 1 > eval_image2text_llava_1.out
python eval_image_to_text.py -m llava -i 2 > eval_image2text_llava_2.out

python eval_image_to_text.py -m gpt4v -i 0 > eval_image2text_gpt4v_0.out
python eval_image_to_text.py -m gpt4v -i 1 > eval_image2text_gpt4v_1.out
python eval_image_to_text.py -m gpt4v -i 2 > eval_image2text_gpt4v_2.out

python eval_image_to_text.py -m gpt4o -i 0 > eval_image2text_gpt4o_0.out
python eval_image_to_text.py -m gpt4o -i 1 > eval_image2text_gpt4o_1.out
python eval_image_to_text.py -m gpt4o -i 2 > eval_image2text_gpt4o_2.out