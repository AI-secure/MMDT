import os
from mmdt.perspectives.fairness.prompts.occupation import occupation_list, prompt_template_neutral, prompt_template_male, prompt_template_female
from mmdt.perspectives.fairness.prompts.occupation import prompt_template_white, prompt_template_black, prompt_template_old, prompt_template_young
from mmdt.perspectives.fairness.prompts.education import education_list,education_prompt_template_neutral,education_prompt_template_male,education_prompt_template_female,education_prompt_template_white,education_prompt_template_black
from mmdt.perspectives.fairness.prompts.daily_activity import activity_list,activity_prompt_template_neutral,activity_prompt_template_male,activity_prompt_template_female
import json

class PromptManager:
    def __init__(self, dataset):
        self.dataset = dataset
        self.prompt_dir = './prompts'

    def get_prompt(self, get_fav_only=False, get_nonfav_only=False):
        prompt_list = []
        prompt_fav = []
        prompt_nonfav = []
        if self.dataset=='test':
            filename = os.path.join(self.prompt_dir, 'test.txt')
            with open(filename, "r") as file:
                for line in file:
                    prompt_list.append(line.strip())
        elif self.dataset=='occupation':
            for job in occupation_list:
                for prompt_template in prompt_template_neutral:
                    prompt_ = prompt_template.replace('[OCCUPATION]', job)
                    prompt_list.append(prompt_)
        elif self.dataset=='occupation_with_sex':
            for job in occupation_list:
                for prompt_template in prompt_template_male:
                    prompt_ = prompt_template.replace('[OCCUPATION]', job)
                    prompt_list.append(prompt_)
                    prompt_fav.append(prompt_)
                for prompt_template in prompt_template_female:
                    prompt_ = prompt_template.replace('[OCCUPATION]', job)
                    prompt_list.append(prompt_)
                    prompt_nonfav.append(prompt_)
        elif self.dataset == 'occupation_with_race':
            for job in occupation_list:
                for prompt_template in prompt_template_white:
                    prompt_ = prompt_template.replace('[OCCUPATION]', job)
                    prompt_list.append(prompt_)
                    prompt_fav.append(prompt_)
                for prompt_template in prompt_template_black:
                    prompt_ = prompt_template.replace('[OCCUPATION]', job)
                    prompt_list.append(prompt_)
                    prompt_nonfav.append(prompt_)
        elif self.dataset == 'occupation_with_age':
            for job in occupation_list:
                for prompt_template in prompt_template_old:
                    prompt_ = prompt_template.replace('[OCCUPATION]', job)
                    prompt_list.append(prompt_)
                    prompt_fav.append(prompt_)
                for prompt_template in prompt_template_young:
                    prompt_ = prompt_template.replace('[OCCUPATION]', job)
                    prompt_list.append(prompt_)
                    prompt_nonfav.append(prompt_)
        elif self.dataset=='education':
            for education in education_list:
                for prompt_template in education_prompt_template_neutral:
                    prompt_ = prompt_template.replace('[EDUCATION]', education)
                    prompt_list.append(prompt_)
        elif self.dataset=='education_with_sex':
            for education in education_list:
                for prompt_template in education_prompt_template_male:
                    prompt_ = prompt_template.replace('[EDUCATION]', education)
                    prompt_list.append(prompt_)
                    prompt_fav.append(prompt_)
                for prompt_template in education_prompt_template_female:
                    prompt_ = prompt_template.replace('[EDUCATION]', education)
                    prompt_list.append(prompt_)
                    prompt_nonfav.append(prompt_)
        elif self.dataset=='education_with_race':
            for education in education_list:
                for prompt_template in education_prompt_template_white:
                    prompt_ = prompt_template.replace('[EDUCATION]', education)
                    prompt_list.append(prompt_)
                    prompt_fav.append(prompt_)
                for prompt_template in education_prompt_template_black:
                    prompt_ = prompt_template.replace('[EDUCATION]', education)
                    prompt_list.append(prompt_)
                    prompt_nonfav.append(prompt_)
        elif self.dataset == 'activity':
            for activity in activity_list:
                for prompt_template in activity_prompt_template_neutral:
                    prompt_ = prompt_template.replace('[ACTIVITY]', activity)
                    prompt_list.append(prompt_)
        elif self.dataset == 'activity_with_sex':
            for activity in activity_list:
                for prompt_template in activity_prompt_template_male:
                    prompt_ = prompt_template.replace('[ACTIVITY]', activity)
                    prompt_list.append(prompt_)
                    prompt_fav.append(prompt_)
                for prompt_template in activity_prompt_template_female:
                    prompt_ = prompt_template.replace('[ACTIVITY]', activity)
                    prompt_list.append(prompt_)
                    prompt_nonfav.append(prompt_)
        else:
            raise ValueError(f'dataset {self.dataset} is not supported')
        if get_fav_only:
            return prompt_fav
        if get_nonfav_only:
            return prompt_nonfav
        return prompt_list

class PromptManager_i2t:
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == "occupation":
            self.prompt_file_path = '/data1/common/mintong/mmdt_results/i2t_images/occupations/occupation_i2t.jsonl'
        elif dataset == "education":
            self.prompt_file_path = '/data1/common/mintong/mmdt_results/i2t_images/education/education_i2t.jsonl'
        elif dataset == "activity":
            self.prompt_file_path = '/data1/common/mintong/mmdt_results/i2t_images/activity/activity_i2t.jsonl'

    def get_prompt(self):

        data = []
        with open(self.prompt_file_path, 'r') as file:
            for line in file:
                entry = json.loads(line)
                data.append(entry)
        return data