import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast

from utils.utils_general import *



def read_langs(file_name, max_line = None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, context_source, column = [], [], [], [], '', []
    max_resp_len = 0


    with open('data/KVR/kvret_entities.json') as f:
        global_entity = json.load(f)
    
    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if '#' in line:
                    context_source += line + '\n'
                    line = line.replace("#","")
                    task_type = line
                    continue

                if task_type == "navigate":
                    column_names = ["@poi", "@type", "@distance", "@traffic_info", "@address"]
                elif task_type == "weather":
                    column_names = ["@location", "@date", "@weather_attribute", "@lowest_temperature","@highest_temperature"]
                elif task_type == "schedule":
                    column_names = ["@event", "@date", "@time", "@party", "@room", "@agenda"]

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    context_source += nid + ' ' + line + '\n'
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid)) 
                    context_arr += gen_u
                    conv_arr += gen_u
                    
                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                    if task_type == "weather": ent_idx_wet = gold_ent
                    elif task_type == "schedule": ent_idx_cal = gold_ent
                    elif task_type == "navigate": ent_idx_nav = gold_ent
                    ent_index = list(set(ent_idx_cal + ent_idx_nav + ent_idx_wet))

                    # Get memory rows pointer labels for dialogue memory, the 1 in the end is for the NULL token
                    selector_index_story = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for
                                            word_arr in context_arr] + [1]


                    # Get memory rows pointer labels for KB memory, the 1 in the end is for the NULL token
                    selector_row_num = np.zeros(len(kb_arr))
                    for ri, kb_row in enumerate(kb_arr):
                        for wi, kb_word in enumerate(kb_row):
                            if kb_word in ent_index or kb_word in r.split():
                                selector_row_num[ri] += 1
                    if len(selector_row_num) > 0:
                        max_num_kb = max(selector_row_num)
                    else:
                        max_num_kb = 1
                    if task_type == 'weather' and max_num_kb >= 3:
                        max_num_kb = 3
                        selector_index_kb = [1 if selector_num >= max_num_kb and selector_num > 0 and kb_arr[irow][0] in ent_index else 0 for irow, selector_num in
                                         enumerate(selector_row_num)] + [1]
                    else:
                        selector_index_kb = [1 if selector_num >= max_num_kb and selector_num > 0 else 0 for selector_num in selector_row_num] + [1]

                    # Get entity pointer position for each word in dialogue memory
                    ptr_index_story = []
                    for key in r.split():
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                        if (index):
                            index = max(index)
                        else:
                            index = len(context_arr)
                        ptr_index_story.append(index)


                    # Get entity pointer position for each word in KB memory
                    ptr_index_kb = []
                    for key in r.split():
                        indexs = [loc for loc, val in enumerate(np.array(kb_arr).flatten()) if (val == key and key in ent_index)]
                        if (indexs):
                            if len(indexs) > 1:
                                max_row = selector_row_num.tolist().index(max(selector_row_num))
                                for index_temp in indexs:
                                    if int(index_temp / len(column_names)) == max_row:
                                        index = index_temp
                                        break
                            else:
                                index = indexs[0]
                        else:
                            index = len(kb_arr) * len(column_names)
                        ptr_index_kb.append(index)



                    # Get sketch responses labels
                    sketch_response = generate_template(global_entity, r, gold_ent, kb_arr, task_type, column)
                    
                    data_detail = {
                        'context_arr':list(context_arr+[['$$$$']*MEM_TOKEN_SIZE]), # $$$$ is NULL token
                        'response':r,
                        'sketch_response':sketch_response,
                        'ptr_index_story':ptr_index_story+[len(context_arr)],
                        'ptr_index_kb': ptr_index_kb + [len(kb_arr)*len(column_names)],
                        'selector_index_story':selector_index_story,
                        'selector_index_kb': selector_index_kb,
                        'ent_index':ent_index,
                        'ent_idx_cal':list(set(ent_idx_cal)),
                        'ent_idx_nav':list(set(ent_idx_nav)),
                        'ent_idx_wet':list(set(ent_idx_wet)),
                        'conv_arr':list(conv_arr),
                        'kb_arr':list(kb_arr+[['$$$$']*MEM_TOKEN_SIZE]),
                        'id':int(sample_counter),
                        'ID':int(cnt_lin),
                        'domain':task_type,
                        'context_source':context_source,
                        'column_arr':column+[['$$$$']*MEM_TOKEN_SIZE]}
                    data.append(data_detail)
                    
                    gen_r = generate_memory(r, "$s", str(nid)) 
                    context_arr += gen_r
                    conv_arr += gen_r
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    r = line
                    context_source += nid + ' ' + line + '\n'
                    kb_info = generate_memory(r, "", str(nid))
                    if task_type == "weather":
                        if kb_info[0] == "today":
                            column.append(["@weekly_time", "@date"] + ["PAD"] * (MEM_TOKEN_SIZE - len(["@weekly_time", "@date"])))
                        else:
                            column.append(column_names)
                    else:
                        column.append(column_names)
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr, context_source, column = [], [], [], '', []
                if(max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_template(global_entity, sentence, sent_ent, kb_arr, domain, kb_column):
    """
    Based on the system response and the provided entity table, the output is the sketch response.
    """
    sketch_response = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                for irow, kb_row in enumerate(kb_arr):
                    for jcol, kb_entity in enumerate(kb_row):
                        if word == kb_entity:
                            ent_type = kb_column[irow][jcol]
                            break
                if ent_type == None:
                    for key in global_entity.keys():
                        if key!='poi':
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                                ent_type = "@" + key
                                break
                        else:
                            poi_list = [d['poi'].lower() for d in global_entity['poi']]
                            if word in poi_list or word.replace('_', ' ') in poi_list:
                                ent_type = "@" + key
                                break
                sketch_response.append(ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response

def generate_memory(sent, speaker, time):

    sent_new = []
    sent_token = sent.split(' ')
    if speaker=="$u" or speaker=="$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn'+str(time), 'word'+str(idx)] + ["PAD"]*(MEM_TOKEN_SIZE-4)
            sent_new.append(temp)
    else:
        sent_token = sent_token + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def prepare_data_seq(batch_size=100):

    if args["dataset"] == 'kvr_navigate':
        file_train = 'data/KVR/navigate_train.txt'
        file_dev = 'data/KVR/navigate_dev.txt'
        file_test = 'data/KVR/navigate_test.txt'
    elif args["dataset"] == 'kvr_schedule':
        file_train = 'data/KVR/schedule_train.txt'
        file_dev = 'data/KVR/schedule_dev.txt'
        file_test = 'data/KVR/schedule_test.txt'
    elif args["dataset"] == 'kvr_weather':
        file_train = 'data/KVR/weather_train.txt'
        file_dev = 'data/KVR/weather_dev.txt'
        file_test = 'data/KVR/weather_test.txt'

    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1
    
    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)
    dev   = get_seq(pair_dev, lang, batch_size, False)
    test  = get_seq(pair_test, lang, batch_size, False)
    
    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))  
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))
    
    return train, dev, test, [], lang, max_resp_len


def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    d = get_seq(pair, lang, batch_size, False)
    return d