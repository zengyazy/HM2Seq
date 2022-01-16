import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast

from utils.utils_general import *


def read_langs(file_name, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, context_source, column, conv_seq = [], [], [], [], '', [], []
    max_resp_len = 0

    with open('data/CamRest/CamRest_entities.json') as f:
        global_entity = json.load(f)
    column_names = ['@name', '@area', '@food', '@phone_number', '@pricerange', "@location", '@address', '@type', '@id', '@postcode']
    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                nid, line = line.split(' ', 1)
                if '\t' in line:
                    context_source += nid + ' ' + line + '\n'
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u
                    conv_seq.append(gen_u)

                    # Get gold entity for each domain
                    ent_index = ast.literal_eval(gold_ent)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    selector_index_story = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in context_arr] + [1]
                    # selector_index_kb = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in kb_arr] + [1]
                    selector_row_num = np.zeros(len(kb_arr))
                    for ri, kb_row in enumerate(kb_arr):
                        for wi, kb_word in enumerate(kb_row):
                            if kb_word in ent_index or kb_word in r.split():
                                selector_row_num[ri] += 1
                    max_num_kb = max(selector_row_num)
                    # if max_num_kb > 2: max_num_kb = 2
                    selector_index_kb = [1 if selector_num >= max_num_kb and selector_num > 0 else 0 for selector_num in
                                             selector_row_num] + [1]

                    # Get local pointer position for each word in system response
                    ptr_index_story = []
                    for key in r.split():
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                        if (index):
                            index = max(index)
                        else:
                            index = len(context_arr)
                        ptr_index_story.append(index)

                    ptr_index_kb = []
                    for key in r.split():
                        indexs = [loc for loc, val in enumerate(np.array(kb_arr).flatten()) if
                                  (val == key and key in ent_index)]
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
                    if len(ptr_index_story) != len(ptr_index_kb):
                        print(
                            "--------------------------len(ptr_index_story) != len(ptr_index_kb)---------------------")
                        os._exit()

                    sketch_response = generate_template(global_entity, r, ent_index, kb_arr, column,
                                                        selector_index_kb)

                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r,
                        'sketch_response': sketch_response,
                        'ptr_index_story': ptr_index_story + [len(context_arr)],
                        'ptr_index_kb': ptr_index_kb + [len(kb_arr) * len(column_names)],
                        'selector_index_story': selector_index_story,
                        'selector_index_kb': selector_index_kb,
                        'ent_index': ent_index,
                        'ent_idx_cal': [],
                        'ent_idx_nav': [],
                        'ent_idx_wet': [],
                        'conv_arr': list(conv_arr),
                        'kb_arr': list(kb_arr + [['$$$$'] * MEM_TOKEN_SIZE]),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': "",
                        'context_source': context_source,
                        'column_arr': column + [['$$$$'] * MEM_TOKEN_SIZE],
                        'conv_seq': list(conv_seq)
                    }
                    # print(data_detail['conv_seq'])
                    # print(len(conv_seq))
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    conv_seq.append(gen_r)

                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    r = line.strip()
                    context_source += nid + ' ' + line + '\n'
                    kb_info = generate_memory(r, "", str(nid))
                    column.append(column_names)

                    # context_arr = kb_info + context_arr
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr, context_source, column, conv_seq = [], [], [], '', [], []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_template(global_entity, sentence, sent_ent, kb_arr, kb_column, selector_index_kb):
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
                        if word == kb_entity and selector_index_kb[irow] == 1:
                            ent_type = kb_column[irow][jcol]
                            # print(ent_type)
                            break
                if ent_type == None:
                    for key in global_entity.keys():
                        if word in global_entity[key]:
                            ent_type = '@' + key
                            break
                if ent_type == None:
                    print(word)
                    print(sentence)
                    print(sent_ent, type(sent_ent))
                    print(global_entity)
                sketch_response.append(ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx)] + ["PAD"] * (MEM_TOKEN_SIZE - 4)
            sent_new.append(temp)
    else:
        sent_token = sent_token + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def prepare_data_seq(batch_size=100):
    file_train = 'data/CamRest/CamRest676_train.txt'
    file_dev = 'data/CamRest/{}CamRest676_dev.txt'
    file_test = 'data/CamRest/{}CamRest676_test.txt'

    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1
    # print(pair_train[0]['conv_seq'])

    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)
    dev = get_seq(pair_dev, lang, batch_size, False)
    test = get_seq(pair_test, lang, batch_size, False)

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
