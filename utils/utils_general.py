import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import numpy as np




def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id, trg_word2id):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        context_arr = self.data_info['context_arr'][index]
        context_arr = self.preprocess(context_arr, self.src_word2id, trg=False)
        response = self.data_info['response'][index]
        response = self.preprocess(response, self.trg_word2id)
        column_arr = self.data_info['column_arr'][index]
        column_arr = self.preprocess(column_arr, self.src_word2id, trg=False)
        # column_index = torch.Tensor(self.data_info['column_index'][index])
        ptr_index_story = torch.Tensor(self.data_info['ptr_index_story'][index])
        ptr_index_kb = torch.Tensor(self.data_info['ptr_index_kb'][index])
        selector_index_story = torch.Tensor(self.data_info['selector_index_story'][index])
        selector_index_kb = torch.Tensor(self.data_info['selector_index_kb'][index])
        conv_arr = self.data_info['conv_arr'][index]
        conv_arr = self.preprocess(conv_arr, self.src_word2id, trg=False)
        kb_arr = self.data_info['kb_arr'][index]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg=False)
        sketch_response = self.data_info['sketch_response'][index]
        sketch_response = self.preprocess(sketch_response, self.trg_word2id)
        
        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        # additional plain information
        data_info['context_arr_plain'] = self.data_info['context_arr'][index]
        data_info['response_plain'] = self.data_info['response'][index]
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][index]
        data_info['sketch_response_plain'] = self.data_info['sketch_response'][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')]+ [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def merge(sequences,story_dim):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths)==0 else max(lengths)

            # if int(args['flag']) == 0 or int(args['flag']) == 2:
            #     MEM_TOKEN_SIZE = 5
            # elif int(args['flag']) == 1:
            #     MEM_TOKEN_SIZE = 6
            # # # print(MEM_TOKEN_SIZE)

            if (story_dim):
                padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    if len(seq) != 0:
                        padded_seqs[i,:end,:] = seq[:end]
            else:
                padded_seqs = torch.ones(len(sequences), max_len).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        def merge_index(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]    
            return padded_seqs, lengths
        
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['conv_arr']), reverse=True) 
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences 
        context_arr, context_arr_lengths = merge(item_info['context_arr'], True)
        response, response_lengths = merge(item_info['response'], False)
        column_arr, _ = merge(item_info['column_arr'], True)
        # column_index, _ = merge_index(item_info['column_index'])
        selector_index_story, _ = merge_index(item_info['selector_index_story'])
        selector_index_kb, _ = merge_index(item_info['selector_index_kb'])
        ptr_index_story, _ = merge(item_info['ptr_index_story'], False)
        ptr_index_kb, _ = merge(item_info['ptr_index_kb'], False)
        conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], True)
        sketch_response, _ = merge(item_info['sketch_response'], False)
        kb_arr, kb_arr_lengths = merge(item_info['kb_arr'], True)
        
        # convert to contiguous and cuda
        context_arr = _cuda(context_arr.contiguous())
        response = _cuda(response.contiguous())
        column_arr = _cuda(column_arr.contiguous())
        selector_index_story = _cuda(selector_index_story.contiguous())
        selector_index_kb = _cuda(selector_index_kb.contiguous())
        ptr_index_story = _cuda(ptr_index_story.contiguous())
        ptr_index_kb = _cuda(ptr_index_kb.contiguous())
        conv_arr = _cuda(conv_arr.transpose(0,1).contiguous())
        sketch_response = _cuda(sketch_response.contiguous())
        # if(len(list(kb_arr.size()))>1): kb_arr = _cuda(kb_arr.transpose(0,1).contiguous())
        
        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        # additional plain information
        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['response_lengths'] = response_lengths
        data_info['conv_arr_lengths'] = conv_arr_lengths
        data_info['kb_arr_lengths'] = kb_arr_lengths

        return data_info


def get_seq(pairs, lang, batch_size, type):
    # if pairs[0]['domain'] == "navigate":
    #     column_names = ["poi", "type", "distance", "traffic_info", "address"]
    # elif pairs[0]['domain'] == "weather":
    #     column_names = ["location", "date", "weather_attribute", "lowest_temperature", "highest_temperature"]
    # else:
    #     column_names = ["event", "date", "time", "party", "room", "agenda"]
    # column_index = torch.zeros(np.array(column_names).shape)
    # for i, column in enumerate(column_names):
    #     lang.index_word(column)
    #     column_index[i] = lang.word2index[column]
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []
    # data_info['column_index'] = []
    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
            # data_info['column_index'].append(column_index)
        if(type):
            lang.index_words(pair['kb_arr'])
            lang.index_words(pair['context_arr'])
            lang.index_words(pair['response'], trg=True)
            lang.index_words(pair['sketch_response'], trg=True)
    lang.index_words(pairs[0]['column_arr'])
    
    dataset = Dataset(data_info, lang.word2index, lang.word2index)
    data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                              batch_size = batch_size,
                                              shuffle = type,
                                              collate_fn = dataset.collate_fn)
    return data_loader
