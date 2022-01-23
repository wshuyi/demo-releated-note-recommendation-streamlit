import streamlit as st
import json
from transformers import AutoTokenizer, AutoModel
import torch
from scipy import spatial
import numpy as np
import pickle
from pathlib import Path


def extract_dict_from_roam_json(myjson, tag_white_list, tag_black_list, keywords_list):
    def handle_block(block, page_title=''):
        if 'children' in block:
            for idx in range(len(block['children'])):
                handle_block(block['children'][idx], page_title)
        showup = False
        if 'string' in block:
            for white_tag in tag_white_list:
                if block['string'].find(f'#{white_tag}')>=0 or block['string'].find(f'[[{white_tag}]]')>=0: #contains the tag
                    showup = True
            for keyword in keywords_list:
                if block['string'].find(keyword)>=0:
                    showup = True
            for black_tag in tag_black_list:
                if block['string'].find(f'#{black_tag}')>=0 or block['string'].find(f'[[{black_tag}]]')>=0: #contains the black tag
                    showup = False
            if showup:
                sample_text = ''
                if 'children' in block:

                    for idx in range(len(block['children'])):
                        sample_text += block['children'][idx]['string']

                
                if not block['uid'] in mydict:
                    mydict[block['uid']] = block['string'] + ' ' + page_title + ' ' + sample_text[:200]


    mydict = dict()

    for idx in range(len(myjson)): #for each page
        page = myjson[idx]
        if 'children' in page:
            for block in page['children']:
                handle_block(block, page['title'])

    return mydict

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



def get_embedding(mystr, tokenizer, model):

    # Sentences we want sentence embeddings for
    sentences = [mystr]

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return np.array(sentence_embeddings[0])



def init_model(local_models):
    if local_models.exists():
        with open(local_models, 'rb') as f:
            model, tokenizer = pickle.load(f)
    else: # no local cache 
    # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        model = AutoModel.from_pretrained('bert-base-chinese')
        with open(local_models, 'wb') as f:
            pickle.dump((model, tokenizer), f)

    
    return tokenizer, model





def get_first_n_similarity(mystr, n, extracted_dict, tokenizer, model):
    def calculate_similarity(myvec, uid):
        vec2 = uid_vect_dict[uid]
        return spatial.distance.cosine(myvec, vec2)
    if len(extracted_dict) > n:
        mydict = dict()
        myvec = get_embedding(mystr, tokenizer, model)

        for uid in extracted_dict:

            mydict[uid] = calculate_similarity(myvec, uid)

        from operator import itemgetter

        res = dict(sorted(mydict.items(), key = itemgetter(1), reverse = False)[:n])
        
        res_dict = dict()
        for uid in list(res.keys()):
            res_dict[uid] = extracted_dict[uid]
        return res_dict
 
    else:
        return extracted_dict


def remove_blank_string_from_list(mylist):
    while("" in mylist) :
        mylist.remove("")
    return mylist


def init_env(mygraph, unhandled_tag_white_list, unhandled_tag_black_list, unhandled_keywords_list, accumulated_extracted_dict):
    
    #check empty strings in lists
    tag_white_list = remove_blank_string_from_list(unhandled_tag_white_list)
    tag_black_list = remove_blank_string_from_list(unhandled_tag_black_list)
    keywords_list = remove_blank_string_from_list(unhandled_keywords_list)

    extracted_dict = extract_dict_from_roam_json(mygraph, tag_white_list, tag_black_list, keywords_list)

    uid_to_embed = []
    for uid in extracted_dict:
        if not uid in uid_vect_dict: #new uid
            uid_to_embed.append(uid)
        elif uid in accumulated_extracted_dict and extracted_dict[uid]!=accumulated_extracted_dict[uid]: # uid content changed
            uid_to_embed.append(uid)
            accumulated_extracted_dict[uid] = extracted_dict[uid] 

    with open(extracted_dict_cache, 'wb') as f:
        pickle.dump(accumulated_extracted_dict, f)
    return uid_to_embed, extracted_dict

def proceed_with_uid_embedding(mystr, n, uid_to_embed, extracted_dict, local_models , uid_vec_cache):
    tokenizer, model = init_model(local_models)
    for uid in uid_to_embed:
        uid_vect_dict[uid] = get_embedding(extracted_dict[uid], tokenizer, model)
    with open(uid_vec_cache, 'wb') as f:
        pickle.dump(uid_vect_dict, f)
    result_dict = get_first_n_similarity(mystr, n, extracted_dict, tokenizer, model)
    st.write(result_dict)


# default_tag_white_list = ['seed', 'active', 'experience', 'zk', '>', 'Readwise']
default_tag_white_list = ['zk']
default_tag_black_list = ['private']
default_keywords_list = []
local_models = Path("local_models.pickle")
roam_json = "/Users/wsy/Dropbox/roam-bak/roamnotes/json/roamwsy.json"
with open(roam_json) as f:
    mygraph = json.load(f)

extracted_dict_cache = "extracted_dict.pickle"
if Path(extracted_dict_cache).exists():
    with open(extracted_dict_cache, 'rb') as f:
        accumulated_extracted_dict = pickle.load(f)
else:
    accumulated_extracted_dict = dict()

uid_vec_cache = "uid_vec_dict.pickle"

if Path(uid_vec_cache).exists():
    with open(uid_vec_cache, 'rb') as f:
        uid_vect_dict = pickle.load(f)
else:
    uid_vect_dict = dict()

default_string = "Python 真是一种非常好的胶水语言，可以把各种模块串联起来，形成合力。脚本写作可以增强系统的自动化啊！"
mystr = st.text_area("这里输入你的新内容：", default_string)
unhandled_tag_white_list = st.text_input("Tags in the white list:", ','.join(default_tag_white_list)).split(',')
unhandled_tag_black_list = st.text_input("Tags in the black list:", ','.join(default_tag_black_list)).split(',')
unhandled_keywords_list = st.text_input("Keywords to extract:", ','.join(default_keywords_list)).split(',')



uid_to_embed = []
extracted_dict = dict()
n = 5
if st.button("Start"):
    
    uid_to_embed, extracted_dict = init_env(mygraph, unhandled_tag_white_list, unhandled_tag_black_list, unhandled_keywords_list, accumulated_extracted_dict)
    st.write(f"Need to get embeddings for {len(uid_to_embed)} uids ...")
    st.button("Just do it!", on_click=proceed_with_uid_embedding, args=[mystr, n, uid_to_embed, extracted_dict, local_models, uid_vec_cache])


