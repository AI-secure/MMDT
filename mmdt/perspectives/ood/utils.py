import torch.nn as nn
import transformers
import torch
import os
import warnings
import clip
import numpy as np
import sklearn.preprocessing
import torch
from packaging import version
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
    
def _get_box_area(obj):
    """
        obj: coordinates of object 1 (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = obj
    w, h = xmax - xmin, ymax - ymin
    area = w * h
    return area


def _check_large(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    a1 = _get_box_area(obj_1)
    a2 = _get_box_area(obj_2)
    if a1 > a2:
        return True
    else:
        return False


def _check_small(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    a1 = _get_box_area(obj_1)
    a2 = _get_box_area(obj_2)
    if a1 < a2:
        return True
    else:
        return False

def _sort_pred_obj(pred_objs, gt_objs):
    """
    Sorting the predicted objects based on the GT objects.
    pred_objs: dict of pred objs. key --> obj_id. val --> cls and cords.
    gt_objs: list of gt cls names.
    """
    sorted_pred_objs = {}
    for key, pred_obj in pred_objs.items():
        if pred_obj['cls'] in gt_objs:
            sorted_pred_objs[gt_objs.index(pred_obj['cls'])] = pred_obj
    return sorted_pred_objs

bigger_words = ["larger", "bigger"]
smaller_words = ["smaller"]
def check_size(results, target, object, size):
    if results["entity_info"][target]["total_count"] == 0 or results["entity_info"][object]["total_count"] == 0:
        return False
    target_box = results['entity_info'][target]["bbox"][0]
    object_box = results['entity_info'][object]["bbox"][0]
    if size in bigger_words:
        return _check_large(target_box, object_box)
    elif size in smaller_words:
        return _check_small(target_box, object_box)
    else:
        return False

def _check_right(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if xmax1 > xmax2:
        return True
    else:
        return False


def _check_left(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if xmin1 < xmin2:
        return True
    else:
        return False


def _check_above(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if (ymin1 < ymin2) or (ymax1 < ymax2):
        return True
    else:
        return False


def _check_below(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if (ymax1 > ymax2) or (ymin1 > ymin2):
        return True
    else:
        return False


def _check_between(obj_1, obj_2, obj_3):
    """
    Check if obj1 in between of obj2 and obj3
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
        obj_3: coordinates of object 3 (xmin, ymin, xmax, ymax)
    """
    # check horizontal dimension:
    if _check_right(obj_1, obj_2) and _check_left(obj_1, obj_3):
        return True
    elif _check_left(obj_1, obj_2) and _check_right(obj_1, obj_3):
        return True
    # check vertical dimension:
    elif _check_below(obj_1, obj_2) and _check_above(obj_1, obj_3):
        return True
    elif _check_above(obj_1, obj_2) and _check_below(obj_1, obj_3):
        return True
    else:
        return False



def check_relation(results, target, objects, relations):
    above_spatial_words = ["on", "above", "over"]
    below_spatial_words = ["below", "beneath", "under"]
    relative_relations = ["between", "among", "in the middle of"]
    if results["entity_info"][target]["total_count"] == 0:
        return False
    target_box = results['entity_info'][target]["bbox"][0]
    object_boxes = []
    for obj in objects:
        if results["entity_info"][obj]["total_count"] == 0:
            return False
        object_boxes.append(results['entity_info'][obj]["bbox"][0])
    if len(objects) == 1:
        if relations[0] == "on the right of":
            if _check_right(target_box, object_boxes[0]):
                return True
        elif relations[0] == "on the left of":
            if _check_left(target_box, object_boxes[0]):
                return True
        elif relations[0] in above_spatial_words:
            if _check_above(target_box, object_boxes[0]):
                return True
        elif relations[0] in below_spatial_words:
            if _check_below(target_box, object_boxes[0]):
                return True
        return False
    elif len(objects) == 2:
        if len(relations) == 2:
            if relations[0] == "on the right of":
                if not _check_right(target_box, object_boxes[0]):
                    return False
            elif relations[0] == "on the left of":
                if not _check_left(target_box, object_boxes[0]):
                    return False
            elif relations[0] in above_spatial_words:
                if not _check_above(target_box, object_boxes[0]):
                    return False
            elif relations[0] in below_spatial_words:
                if not _check_below(target_box, object_boxes[0]):
                    return False
            
            if relations[1] == "on the right of":
                if _check_right(target_box, object_boxes[1]):
                    return True
            elif relations[1] == "on the left of":
                if _check_left(target_box, object_boxes[1]):
                    return True
            elif relations[1] in above_spatial_words:
                if _check_above(target_box, object_boxes[1]):
                    return True
            elif relations[1] in below_spatial_words:
                if _check_below(target_box, object_boxes[1]):
                    return True
        else:
            if _check_between(target_box, object_boxes[0], object_boxes[1]):
                return True
    elif len(objects) == 3:
        if len(relations) == 2:
            if relations[0] == "on the right of":
                if not _check_right(target_box, object_boxes[1]) and not _check_right(object_boxes[0], object_boxes[1]):
                    return False
            elif relations[0] == "on the left of":
                if not _check_left(target_box, object_boxes[1]) and not _check_left(object_boxes[0], object_boxes[1]):
                    return False
            elif relations[0] in above_spatial_words:
                if not _check_above(target_box, object_boxes[1]) and not _check_above(object_boxes[0], object_boxes[1]):
                    return False
            elif relations[0] in below_spatial_words:
                if not _check_below(target_box, object_boxes[1]) and not _check_below(object_boxes[0], object_boxes[1]):
                    return False
            if relations[1] == "on the right of":
                if _check_right(target_box, object_boxes[2]) and _check_right(object_boxes[0], object_boxes[2]):
                    return True
            elif relations[1] == "on the left of":
                if _check_left(target_box, object_boxes[2]) and _check_left(object_boxes[0], object_boxes[2]):
                    return True
            elif relations[1] in above_spatial_words:
                if _check_above(target_box, object_boxes[2]) and _check_above(object_boxes[0], object_boxes[2]):
                    return True
            elif relations[1] in below_spatial_words:
                if _check_below(target_box, object_boxes[2]) and _check_below(object_boxes[0], object_boxes[2]):
                    return True
        else:
            if _check_between(target_box, object_boxes[1], object_boxes[2]) and _check_between(object_boxes[0], object_boxes[1], object_boxes[2]):
                return True
    return False

def cal_spatial_acc(data):
    true_count = 0
    refusal = 0
    for img_id, sample in data.items():
        if "error" in sample:
            refusal += 1
            continue
        if sample["objects"]["answer"] == "Wrong":
            continue
        else:
            true_count += 1
    acc = 100 * (true_count / len(data.keys()))
    return acc

def cal_acc(data):
    true_count = 0
    for img_id, sample in data.items():
        if "error" in sample:
            continue
        for item_id, item in enumerate(sample["objects"]):
            if item["answer"] == "Wrong":
                break
            elif item_id == len(sample["objects"]) - 1:
                true_count += 1
    acc = 100 * (true_count / len(data.keys()))
    return acc

def cal_counting_acc(data):
    # Calculate the Acc:
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for img_id, sample in data.items():
        for item_id, item in enumerate(sample["objects"]):
            pred_num = 0
            gt_num = int(item["counts"])
            pred_num = int(item["generated_objects"])
            true_pos += min(gt_num, pred_num)
            false_pos = false_pos + max((pred_num-gt_num), 0)
            false_neg = false_neg + max((gt_num-pred_num), 0)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, append=False, prefix='A photo depicts'):
        self.data = data
        self.prefix = ''
        if append:
            self.prefix = prefix
            if self.prefix[-1] != ' ':
                self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


def Convert(image):
    return image.convert("RGB")


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8, append=False):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions, append=append),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, append=False, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device, append=append)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
        candidates = candidates / \
            np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def clipeval_with_index(image_dir, target_prompts, index, device):
    image_ids = index
    image_paths = [os.path.join(image_dir, f"{i}.png") for i in index]

    print(image_paths[:5])

    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    clipscores = []
    clipscores_std = []
    clipscores_all = []
    append = True

    _, per_instance_image_text, _ = get_clip_score(
        model, image_feats, target_prompts, device, append=append)
    scores_each_prompt = {image_id: {'CLIPScore': float(clipscore)}
                            for image_id, clipscore in
                            zip(image_ids, per_instance_image_text)}

    clipscores.append(np.mean([s['CLIPScore'] for s in scores_each_prompt.values()]))
    clipscores_std.append(np.std([s['CLIPScore'] for s in scores_each_prompt.values()]))
    clipscores_all.append(np.array([s['CLIPScore'] for s in scores_each_prompt.values()]))

    print("std:", clipscores_std)
    return scores_each_prompt, clipscores_all



def calclipscore_with_index(prompt,  outpath, index):
    device = 'cuda'
    outdir = outpath
    clip_scores, clip_score_all=clipeval_with_index(str(outdir), prompt, index, device)
    print("Clip score mean: {}, Clip score std: {}".format(np.mean(clip_score_all), np.std(clip_score_all)))
    return  clip_scores, clip_score_all






class LLMChat(nn.Module):
    def __init__(self, model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'):
        super(LLMChat, self).__init__()
        self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.float16},
                device="cuda",
            )

    def forward(self, prompt, max_new_tokens = 500):
        messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
        
        prompt = self.pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,
        )
        
        return outputs[0]["generated_text"][len(prompt):]
