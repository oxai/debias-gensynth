import argparse
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import pandas as pd
import numpy as np 
import math
import os
import tqdm

# Define the gender words list
MALE_GENDERED_WORDS = ["man", "boy", "male", "gentleman", "his", "he", "him", "son", "father", "brother", "uncle", "nephew", "husband", "boys", "men", "gentlemen", "sons", "fathers", "brothers", "uncles", "nephews", "husbands"]
FEMALE_GENDERED_WORDS = ["woman", "girl", "female", "lady", "her", "she", "hers", "daughter", "mother", "sister", "aunt", "niece", "wife", "girls", "women", "ladies", "daughters", "mothers", "sisters", "aunts", "nieces", "wives"]

FEMALE_TO_MALE = {"man": "woman", "boy": "girl", "male": "female", "gentleman": "lady", "his": "her", "he": "she", "him": "hers", "son": "daughter", "father": "mother", "brother": "sister", "uncle": "aunt", "nephew": "niece", "husband": "wife", "boys": "girls", "men": "women", "gentlemen": "ladies", "sons": "daughters", "fathers": "mothers", "brothers": "sisters", "uncles": "aunts", "nephews": "nieces", "husbands": "wifes"}
MALE_TO_FEMALE = {"woman": "man", "girl": "boy", "female": "male", "lady": "gentleman", "her": "his", "she": "he", "hers": "him", "daughter": "son", "mother": "father", "sister": "brother", "aunt": "uncle", "niece": "nephew", "wife": "husband", "girls": "boys", "women": "men", "ladies": "gentlemen", "daughters": "sons", "mothers": "fathers", "sisters": "brothers", "aunts": "uncles", "nieces": "nephews", "wives": "husbands"}

MALE_GENDERED_BANK = {"man": "person", "boy": "child", "male": "person", "gentleman": "person", "his": "their", "he": "they", "him": "them", "son": "child", "father": "parent", "brother": "sibling", "uncle": "relative", "nephew": "relative", "husband": "partner", "boys": "children", "men": "people", "gentlemen": "people", "sons": "children", "fathers": "parents", "brothers": "siblings", "uncles": "relatives", "nephews": "relatives", "husbands": "partners"}
FEMALE_GENDERED_BANK = {"woman": "person", "girl": "child", "female": "person", "lady": "person", "her": "their", "she": "they", "hers": "theirs", "daughter": "child", "mother": "parent", "sister": "sibling", "aunt": "relative", "niece": "relative", "wife": "partner", "girls": "children", "women": "people", "ladies": "people", "daughters": "children", "mothers": "parents", "sisters": "siblings", "aunts": "relatives", "nieces": "relatives", "wives": "partners"}

# add the two banks
BANK_COMBINED = {**MALE_GENDERED_BANK, **FEMALE_GENDERED_BANK}

def remove_gender_words(sentence, gender_words):
    for word in gender_words:
        sentence = remove_word(sentence, word)
    return sentence

def replace_gender_words(sentence, gender_dict):
    for key, val in gender_dict.items():
        sentence = replace_words(sentence, key, val)
    return sentence

def remove_word(sentence: str, target_word: str) -> str:
    # Wrap the target word with word boundaries to avoid partial match
    target = fr"\b{target_word}\b"

    # Remove the target word
    sentence = re.sub(target, "", sentence)

    # Handle extra spaces and punctuation
    sentence = re.sub(r'\s+', ' ', sentence)  # remove double spaces
    sentence = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', sentence)  # remove space before punctuation

    return sentence.strip()  # remove leading and trailing spaces

def replace_words(sentence: str, target_word: str, replacement_word: str) -> str:
    # replace the target word with replacement word only if the target word is not part of another word
    target = fr"\b{target_word}\b"
    sentence = re.sub(target, replacement_word, sentence)

    # Handle extra spaces and punctuation
    sentence = re.sub(r'\s+', ' ', sentence)  # remove double spaces
    sentence = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', sentence)  # remove space before punctuation

    return sentence.strip()  # remove leading and trailing spaces

def mask_gender_words(sentence, gender_words):
    for word in gender_words:
        sentence = re.sub(r'\b' + word + r'\b', '[MASK]', sentence, flags=re.IGNORECASE)
    return sentence

def assign_gender_label(sentence, male_words, female_words):
    # Convert the sentence to lowercase and split into words
    words = set(sentence.lower().split())

    # Check for the presence of male and female words in the sentence
    male_words_present = any(word in words for word in male_words)
    female_words_present = any(word in words for word in female_words)

    # Assign a gender label based on the presence of male and female words
    if male_words_present and not female_words_present:
        return 0  # Male
    elif female_words_present and not male_words_present:
        return 1  # Female
    elif male_words_present and female_words_present:
        return 2
    else:
        return -1

def bias_at_k(similarities, labels, bias_k_arr=(1,5,10), skew_k_arr=(25, 100)):
    labels = pd.Series(labels)
    similarities.shape[0] == len(labels)
    topk_indices = similarities.argsort()[::-1]

    metrics = {}
    for k in bias_k_arr:
        topk_ = topk_indices[:k]
        topk_labels = labels[topk_]
        gcounts = topk_labels.value_counts()
        if len(gcounts) == 1 and gcounts.index[0] == -1:
            metrics["bias@{}".format(k)] = 0
        else:
            num_men = gcounts[0] if 0 in gcounts.index else 0
            num_women = gcounts[1] if 1 in gcounts.index else 0
            metrics["bias@{}".format(k)] = (num_men - num_women) / (num_women + num_men)

    # remove -1s from labels and similarities
    similarities = similarities[labels != -1]
    labels = labels[labels != -1].reset_index(drop=True)
    topk_indices = similarities.argsort()[::-1]
    
    gcounts = labels.value_counts()
    demographic_parities = [gcounts[0] / gcounts.sum(), gcounts[1] / gcounts.sum()]

    for k in skew_k_arr:
        topk_ = topk_indices[:k]
        topk_labels = labels[topk_]
        gcounts = topk_labels.value_counts()

        skews = []
        for lab in gcounts.index:
        # for lab, dem_parity in enumerate(demographic_parities):
            dem_parity = demographic_parities[lab]
            lab_dst = gcounts[lab] / gcounts.sum()
            skew = math.log(lab_dst / dem_parity)
            skews.append(skew)
        metrics["maxskew@{}".format(k)] = max(skews)

    return metrics

def load_coco_caps(data_dir, split="val", balanced=False):
    # load coco captions
    COCO_anno_fn = os.path.join(data_dir, "annotations", "captions_{}2017.json".format(split))
    with open(COCO_anno_fn, "r") as f:
        val_data = json.load(f)['annotations']
    val_data = pd.DataFrame(val_data)

    # assign gender labels to each caption (-1: neutral, 0: male, 1: female, 2: both)
    val_data['gender'] = val_data['caption'].apply(lambda x: assign_gender_label(x, MALE_GENDERED_WORDS, FEMALE_GENDERED_WORDS))

    # assign gender label to each iamge
    image_labels = {}
    for image_id in val_data['image_id'].unique():
        vdf = val_data[val_data['image_id'] == image_id]
        gcounts = vdf['gender'].value_counts()
        label = -1
        if 2 in gcounts:
            label = -1
        elif 0 in gcounts and 1 not in gcounts:
            label = 0
        elif 1 in gcounts and 0 not in gcounts:
            label = 1
        image_labels[image_id] = label

    image_labels = pd.Series(image_labels)
    val_data.set_index('image_id', inplace=True)
    val_data['gender'] = image_labels
    val_data.reset_index(inplace=True)

    print("COCO unbalanced: ", val_data.drop_duplicates('image_id')['gender'].value_counts())
    if balanced:
        # sub-sample to ensure number of gender 0s and 1s are equal
        # Create a random DataFrame for the demonstration
        # Count the occurrences of 0 and 1
        unique_img_df = val_data.drop_duplicates(subset=['image_id'])
        count_0 = (unique_img_df['gender'] == 0).sum()
        count_1 = (unique_img_df['gender'] == 1).sum()
        count_minus1 = (unique_img_df['gender'] == -1).sum()
        frac_minus1 = count_minus1 / (count_0 + count_1 + count_minus1)

        # Decide which group is less frequent
        less_freq_val = 0 if count_0 < count_1 else 1
        more_freq_val = 1 if less_freq_val == 0 else 0

        # Sample from the more frequent group to match the count of the less frequent one
        df_more_freq_sampled = unique_img_df[unique_img_df['gender'] == more_freq_val].sample(count_0 if less_freq_val == 0 else count_1)

        # Concatenate the less frequent group and the sampled data from the more frequent one
        df_balanced = pd.concat([unique_img_df[unique_img_df['gender'] == less_freq_val], df_more_freq_sampled])

        # val_data = df_balanced
        # If you want to include -1, then add this line
        
        minus1_df = unique_img_df[unique_img_df['gender'] == -1]
        # keep original proportion of -1s the same
        minus1_df = minus1_df.sample(int(frac_minus1 * len(df_balanced) / (1-frac_minus1)))

        unique_img_df = pd.concat([minus1_df, df_balanced])

        print("COCO balanced: ")
        print(unique_img_df['gender'].value_counts())
        val_data = val_data[val_data['image_id'].isin(unique_img_df['image_id'].values)]

    val_data.reset_index(inplace=True, drop=True)
    return val_data


def main(args):
    # load coco captions
    val_data = load_coco_caps(args.data_dir, split="val", balanced=args.balanced_coco)

    # convert strings to lower case
    val_data['caption'] = val_data['caption'].apply(lambda x: x.lower())

    # post-process captions
    if args.model == "tfidf":
        val_data['caption'] = val_data['caption'].apply(lambda x: remove_gender_words(x, MALE_GENDERED_WORDS + FEMALE_GENDERED_WORDS))
    elif args.model == "bert":
        val_data['caption'] = val_data['caption'].apply(lambda x: mask_gender_words(x, MALE_GENDERED_WORDS + FEMALE_GENDERED_WORDS))
    elif args.model == "clip":
        val_data['caption'] = val_data['caption'].apply(lambda x: replace_gender_words(x, BANK_COMBINED))
    elif args.model == "clip-clip100":
        val_data['caption'] = val_data['caption'].apply(lambda x: replace_gender_words(x, BANK_COMBINED))
    elif args.model == "debias-clip":
        val_data['caption'] = val_data['caption'].apply(lambda x: replace_gender_words(x, BANK_COMBINED))
    elif args.model == "random":
        pass
    else:
        raise ValueError("Invalid model type")
    image_df = val_data.drop_duplicates(subset=['image_id'])
    corpus_final = val_data['caption'].values.tolist()


    # LOAD FEATURES
    if args.model == "random":
        import torch
        print("Generating random tensors...")
        text_feats = torch.rand(len(corpus_final), 512).numpy()
        text_feats = text_feats / np.linalg.norm(text_feats, axis=1)[:, None]
        img_feats = torch.rand(len(image_df['image_id'].unique()), 512).numpy()
        img_feats = img_feats / np.linalg.norm(img_feats, axis=1)[:, None]
    elif args.model == "tfidf":
        # TF-IDF on captions
        tfidf_vectorizer = TfidfVectorizer()
        text_feats = tfidf_vectorizer.fit_transform(corpus_final)
        img_feats = text_feats
    elif args.model == "bert":
        from transformers import BertTokenizer, BertModel
        import torch
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased", num_labels=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        corpus_feats = []
        # TODO this could be improved with batching / pipeline
        print("Extracting BERT features...")
        for cdx, caption in tqdm.tqdm(enumerate(corpus_final), total=len(corpus_final)):
            inputs = tokenizer(caption, return_tensors="pt", max_length=128, truncation=True)
            inputs.to(device)
            outputs = model(**inputs)
            last_hidden_states = outputs[0]
            last_hidden_states = last_hidden_states.detach().cpu().numpy()
            corpus_feats.append(last_hidden_states[0][0])
        corpus_feats = np.array(corpus_feats)
        # normalize
        text_feats = corpus_feats / np.linalg.norm(corpus_feats, axis=1)[:, None]
        img_feats = text_feats
    elif args.model == "clip":
        from transformers import CLIPProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
        import torch
        from PIL import Image
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        corpus_feats = []
        # TODO this could be sped up with batching / pipeline
        print("Extracting CLIP features...")
        for cdx, caption in tqdm.tqdm(enumerate(corpus_final), total=len(corpus_final)):
            inputs = processor(text=[caption], return_tensors="pt", padding=True)
            inputs.to(device)
            outputs = model(**inputs)
            last_hidden_states = outputs[0]
            last_hidden_states = last_hidden_states.detach().cpu().numpy()
            corpus_feats.append(last_hidden_states[0])
        corpus_feats = np.array(corpus_feats)
        # normalize
        text_feats = corpus_feats / np.linalg.norm(corpus_feats, axis=1)[:, None]

        img_feats = []
        del model
        model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        # TODO this could be sped up with batching / pipeline
        print("Extracting CLIP image features...")
        for cdx, image_id in tqdm.tqdm(enumerate(image_df['image_id'].unique()), total=len(image_df['image_id'].unique())):
            image_fp = os.path.join(args.data_dir, "images", "val2017", "{:012d}.jpg".format(image_id))
            image = Image.open(image_fp)
            inputs = processor(images=image, return_tensors="pt")
            inputs.to(device)
            outputs = model(**inputs)
            last_hidden_states = outputs[0]
            last_hidden_states = last_hidden_states.detach().cpu().numpy()
            img_feats.append(last_hidden_states[0])
        img_feats = np.array(img_feats)
        # normalize
        img_feats = img_feats / np.linalg.norm(img_feats, axis=1)[:, None]
    elif args.model == "clip-clip100":
        from transformers import CLIPProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
        import torch
        from PIL import Image
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # dim_clip_indexes100 = np.load("/home/mlfarinha/vision-lang-bias-mt22/bias-k/clip_clip_m100_idx.npy")
        dim_clip_indexes100 = [366, 344, 422, 331, 424, 52, 51, 324, 313, 427, 304, 435, 440, 442, 82, 134, 417, 415, 96, 411, 81, 382, 383, 385, 84, 386, 391, 445, 370, 176, 90, 400, 402, 365, 362, 94, 72, 446, 129, 291, 248, 236, 232, 215, 13, 212, 489, 448, 492, 207, 501, 6, 195, 191, 507, 187, 495, 472, 379, 267, 287, 268, 449, 451, 23, 469, 269, 273, 143, 28, 460, 257, 280, 141, 307, 125, 97, 182, 456, 480, 369, 89, 353, 336, 180, 487, 468, 122, 130, 393, 342, 341, 167, 466, 404, 508, 62, 36, 256, 49, 358, 105, 116, 241, 337, 190, 107, 138, 334, 343, 205, 29, 296, 172, 332, 254, 509, 77, 171, 131, 395, 204, 8, 150, 103, 226, 426, 240, 413, 92, 477, 170, 48, 127, 340, 271, 399, 467, 35, 21, 463, 221, 124, 233, 54, 310, 15, 119, 303, 431, 0, 109, 471, 505, 64, 20, 41, 444, 443, 61, 133, 5, 377, 14, 356, 308, 419, 282, 297, 71, 128, 227, 73, 318, 126, 55, 65, 488, 255, 433, 79, 213, 490, 24, 418, 44, 259, 185, 117, 409, 30, 253, 132, 194, 57, 189, 166, 144, 32, 360, 161, 252, 277, 292, 374, 281, 115, 371, 66, 276, 363, 349, 483, 203, 439, 33, 238, 352, 405, 491, 168, 158, 294, 478, 108, 262, 242, 19, 175, 406, 503, 11, 345, 270, 7, 160, 18, 438, 47, 101, 120, 193, 118, 214, 80, 22, 412, 219, 375, 328, 234, 58, 283, 98, 206, 278, 114, 298, 394, 388, 407, 74, 3, 289, 231, 312, 1, 9, 496, 338, 354, 434, 45, 186, 70, 26, 147, 272, 88, 69, 396, 246, 258, 311, 228, 110, 164, 314, 397, 368, 169, 46, 359, 60, 183, 104, 420, 235, 201, 244, 50, 225, 37, 511, 260, 389, 137, 251, 135, 295, 458, 392, 34, 348, 447, 325, 217, 40, 243, 56, 497, 301, 198, 220, 197, 123, 510, 177, 309, 93, 493, 414, 68, 455, 102, 237, 347, 25, 95, 249, 200, 12, 91, 274, 43, 229, 181, 196, 157, 216, 31, 429, 479, 288, 145, 500, 199, 192, 486, 387, 266, 350, 475, 155, 285, 339, 59, 502, 462, 320, 372, 211, 250, 306, 473, 284, 464, 485, 159, 293, 156, 100, 361, 465, 279, 499, 333, 476, 474, 78, 457, 223, 247, 506, 42, 346, 323, 27, 142, 401, 302, 452, 315, 504, 275, 403, 329, 87, 459, 436, 290, 63]
        
        corpus_feats = []
        # TODO this could be sped up with batching / pipeline
        print("Extracting CLIP-CLIP (m=100) features...")
        for cdx, caption in tqdm.tqdm(enumerate(corpus_final), total=len(corpus_final)):
            inputs = processor(text=[caption], return_tensors="pt", padding=True)
            inputs.to(device)
            outputs = model(**inputs)
            last_hidden_states = outputs[0]
            last_hidden_states = last_hidden_states.detach().cpu().numpy()
            last_hidden_states = last_hidden_states[:, dim_clip_indexes100]
            corpus_feats.append(last_hidden_states[0])
        corpus_feats = np.array(corpus_feats)
        # normalize
        text_feats = corpus_feats / np.linalg.norm(corpus_feats, axis=1)[:, None]

        img_feats = []
        del model
        model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        # TODO this could be sped up with batching / pipeline
        print("Extracting CLIP-CLIP (m=100) image features...")
        for cdx, image_id in tqdm.tqdm(enumerate(image_df['image_id'].unique()), total=len(image_df['image_id'].unique())):
            image_fp = os.path.join(args.data_dir, "images", "val2017", "{:012d}.jpg".format(image_id))
            image = Image.open(image_fp)
            inputs = processor(images=image, return_tensors="pt")
            inputs.to(device)
            outputs = model(**inputs)
            last_hidden_states = outputs[0]
            last_hidden_states = last_hidden_states.detach().cpu().numpy()
            last_hidden_states = last_hidden_states[:, dim_clip_indexes100]
            img_feats.append(last_hidden_states[0])
        img_feats = np.array(img_feats)
        # normalize
        img_feats = img_feats / np.linalg.norm(img_feats, axis=1)[:, None]
    elif args.model == "debias-clip":
        from transformers import CLIPProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
        import debias_clip
        import clip
        import torch
        from PIL import Image
        # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = debias_clip.load("ViT-B/16-gender", device=device)
        model.to(device)

        corpus_feats = []
        # TODO this could be sped up with batching / pipeline
        print("Extracting Debias CLIP features...")
        for cdx, caption in tqdm.tqdm(enumerate(corpus_final), total=len(corpus_final)):
            inputs = clip.tokenize([caption]).to(device)
            outputs = model.encode_text(inputs)
            outputs = outputs[0].detach().cpu().numpy()
            corpus_feats.append(outputs)
        corpus_feats = np.array(corpus_feats)
        # normalize
        text_feats = corpus_feats / np.linalg.norm(corpus_feats, axis=1, keepdims=True)

        img_feats = []
        del model
        model, preprocess = debias_clip.load("ViT-B/16-gender", device=device)
        model.to(device)
        # TODO this could be sped up with batching / pipeline
        print("Extracting Debias CLIP image features...")
        for cdx, image_id in tqdm.tqdm(enumerate(image_df['image_id'].unique()), total=len(image_df['image_id'].unique())):
            image_fp = os.path.join(args.data_dir, "images", "val2017", "{:012d}.jpg".format(image_id))
            image = Image.open(image_fp)
            inputs = torch.unsqueeze(preprocess(image), dim=0).to(device)
            outputs = model.encode_image(inputs)
            outputs = outputs[0].detach().cpu().numpy()
            img_feats.append(outputs)
        img_feats = np.array(img_feats)
        # normalize
        img_feats = img_feats / np.linalg.norm(img_feats, axis=1, keepdims=True)
    else:
        raise ValueError("Invalid model type")

    # COMPUTE BIAS RESULTS
    bias_k_arr = []
    for vdx, vec in tqdm.tqdm(enumerate(text_feats), total=len(corpus_final)):
        if args.model == "random":
            cosine_similarities = linear_kernel(vec.reshape(1, -1), img_feats).flatten()
        elif args.model == "tfidf":
            cosine_similarities = linear_kernel(vec, img_feats).flatten()
        elif args.model == "bert":
            cosine_similarities = linear_kernel(vec.reshape(1, -1), img_feats).flatten()
        elif args.model == "clip":
            cosine_similarities = linear_kernel(vec.reshape(1, -1), img_feats).flatten()
        elif args.model == "clip-clip100":
            cosine_similarities = linear_kernel(vec.reshape(1, -1), img_feats).flatten()
        elif args.model == "debias-clip":
            cosine_similarities = linear_kernel(vec.reshape(1, -1), img_feats).flatten()
        else:
            raise ValueError("Invalid model type")
            
        if args.model in ["tfidf", "bert"]:
            # FairModel eval, remove captions with same IDS
            labels = val_data['gender'].values
            image_id = val_data.iloc[vdx]['image_id']
            # indices with same image id
            same_image_indices = val_data[val_data['image_id'] == image_id].index.tolist()
            # # remove indices from cosine similarities
            cosine_similarities = np.delete(cosine_similarities, same_image_indices)
            # # remove indices from labels
            labels = np.delete(val_data['gender'].values, same_image_indices)
        elif args.model in ["random", "clip", "clip-clip100", "debias-clip"]:
            labels = image_df["gender"].values
        metrics = bias_at_k(cosine_similarities, labels)
        bias_k_arr.append(metrics)

    bias_k_arr = pd.DataFrame(bias_k_arr)
    # take mean of each column
    bias_k_arr_m = bias_k_arr.mean(axis=0)
    print(bias_k_arr_m)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert", choices=["bert", "random", "tfidf", "clip", "clip-clip100", "debias-clip"], help="model type")
    parser.add_argument("--balanced_coco", action="store_true", help="whether to use balanced coco dataset")
    parser.add_argument("--data_dir", default="/tmp/COCO2017", type=str, help="data directory")

    args = parser.parse_args()

    main(args)