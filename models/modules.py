import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.utils_general import _cuda


class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(ContextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.W = nn.Linear(2*hidden_size, hidden_size)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(1, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))
        embedded = torch.sum(embedded, 2).squeeze(2)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = hidden[0]
        return outputs.transpose(0,1), hidden

class ExternalKnowledge(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 
        for hop in range(self.max_hops+1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("CS_{}".format(hop), C)
        self.CS = AttrProxy(self, "CS_")
        for hop in range(self.max_hops+1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("CK_{}".format(hop), C)
        self.CK = AttrProxy(self, "CK_")
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.W = nn.Linear(embedding_dim*2, embedding_dim)
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)

    def add_lm_embedding(self, full_memory, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            end = conv_len[bi]
            full_memory[bi, :end, :] = full_memory[bi, :end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def load_memory(self, story, kb, kb_columnm, kb_len, conv_len, hidden, dh_outputs):
        # Forward multiple hop mechanism
        u_story = [hidden.squeeze(0)]
        story_size = story.size()
        kb_size = kb.size()
        u_kb = [hidden.squeeze(0)]
        self.m_story = []
        self.m_kb = []
        for hop in range(self.max_hops):
            embed_A_story = self.CS[hop](story.contiguous().view(story_size[0], -1))#.long()) # b * (m * s) * e
            embed_A_story = embed_A_story.view(story_size+(embed_A_story.size(-1),)) # b * m * s * e
            embed_A_story = torch.sum(embed_A_story, 2).squeeze(2) # b * m * e
            embed_A_story = self.add_lm_embedding(embed_A_story, conv_len, dh_outputs)
            embed_A_story = self.dropout_layer(embed_A_story)
            
            if(len(list(u_story[-1].size()))==1):
                u_story[-1] = u_story[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u_story[-1].unsqueeze(1).expand_as(embed_A_story)
            story_prob_logit = torch.sum(embed_A_story*u_temp, 2)
            story_prob_   = self.softmax(story_prob_logit)
            
            embed_C_story = self.CS[hop+1](story.contiguous().view(story_size[0], -1).long())
            embed_C_story = embed_C_story.view(story_size+(embed_C_story.size(-1),))
            embed_C_story = torch.sum(embed_C_story, 2).squeeze(2)
            embed_C_story = self.add_lm_embedding(embed_C_story, conv_len, dh_outputs)

            story_prob = story_prob_.unsqueeze(2).expand_as(embed_C_story)
            o_k  = torch.sum(embed_C_story*story_prob, 1)
            u_k = u_story[-1] + o_k
            u_story.append(u_k)
            self.m_story.append(embed_A_story)
        self.m_story.append(embed_C_story)
        for hop in range(self.max_hops):
            embed_column = self.CK[hop](kb_columnm.contiguous()) #b * m * s * e
            embed_A_kb = self.CK[hop](kb.contiguous().view(kb_size[0], -1))  # .long()) # b * (m * s) * e
            embed_A_kb = embed_A_kb.view(kb_size + (embed_A_kb.size(-1),))  # b * m * s * e

            embed_A_kb = embed_A_kb + embed_column  # b * m * s * e
            embed_A_kb = torch.sum(embed_A_kb, dim=2).squeeze(2) # b * m * e

            if (len(list(u_kb[-1].size())) == 1):
                u_kb[-1] = u_kb[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u_kb[-1].unsqueeze(1).expand_as(embed_A_kb)
            kb_prob_logit = torch.sum(embed_A_kb * u_temp, 2)
            kb_prob_ = self.softmax(kb_prob_logit)


            embed_column = self.CK[hop+1](kb_columnm.contiguous())  # b * m * s * e
            embed_C_kb = self.CK[hop + 1](kb.contiguous().view(kb_size[0], -1).long())
            embed_C_kb = embed_C_kb.view(kb_size + (embed_C_kb.size(-1),))
            embed_C_kb = embed_C_kb + embed_column  # b * m * s * e
            embed_C_kb = torch.sum(embed_C_kb, dim=2) # b * m * e


            kb_prob = kb_prob_.unsqueeze(2).expand_as(embed_C_kb) #b * m  * e
            o_k = torch.sum(embed_C_kb * kb_prob, 1) # b*e
            u_k = u_story[-1] + o_k
            u_kb.append(u_k)
            self.m_kb.append(embed_A_kb)
        self.m_kb.append(embed_C_kb)
        read_out = u_kb[-1]
        return self.sigmoid(story_prob_logit), self.sigmoid(kb_prob_logit), read_out

    def forward(self, query_vector, global_pointer, story=True):
        u = [query_vector]
        for hop in range(self.max_hops):
            if story:
                m_A = self.m_story[hop]
            else:
                m_A = self.m_kb[hop]
            m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)  #b * m * e

            if(len(list(u[-1].size()))==1): 
                u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.

            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A * u_temp, 2)
            prob_soft = self.softmax(prob_logits)

            if story:
                m_C = self.m_story[hop+1]
            else:
                m_C = self.m_kb[hop+1]
            m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits


class LocalMemoryDecoder(nn.Module):
    def __init__(self, shared_emb, lang, embedding_dim, hop, dropout):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 
        self.C = shared_emb 
        self.softmax = nn.Softmax(dim=1)
        self.sketch_rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(2*embedding_dim, embedding_dim)
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, extKnow, column_arr, story_size, kb_size, story_lengths, kb_lengths, copy_list_story, copy_list_kb, encode_hidden, target_batches, max_target_length, batch_size, use_teacher_forcing, get_decoded_words, story_global_pointer, kb_global_pointer):
        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_decoder_outputs_ptr_story = _cuda(torch.zeros(max_target_length, batch_size, story_size[1]))
        all_decoder_outputs_ptr_kb = _cuda(torch.zeros(max_target_length, batch_size, kb_size[1]*kb_size[2]))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        memory_mask_for_step_story = _cuda(torch.ones(story_size[0], story_size[1]))
        memory_mask_for_step_kb = _cuda(torch.ones(kb_size[0], kb_size[1], kb_size[2]))
        decoded_fine, decoded_coarse = [], []
        gate = []
        probability_kb = []
        probability_story = []
        
        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)
        embed_column = self.dropout_layer(self.C(column_arr)) #b * m * s * e
        column_score_view = []
        
        # Start to generate word-by-word
        for t in range(max_target_length):
            embed_q = self.dropout_layer(self.C(decoder_input)) # b * e
            if len(embed_q.size()) == 1: embed_q = embed_q.unsqueeze(0)
            _, hidden = self.sketch_rnn(embed_q.unsqueeze(0), hidden)
            query_vector = hidden[0]

            p_vocab = self.attend_vocab(self.C.weight, hidden.squeeze(0))
            all_decoder_outputs_vocab[t] = p_vocab
            _, topvi = p_vocab.data.topk(1)
            
            # query the dialogue memory using the hidden state of sketch RNN
            story_prob_soft, story_prob_logits = extKnow(query_vector, story_global_pointer, True)
            all_decoder_outputs_ptr_story[t] = story_prob_logits

            # Calculate the probability that the value of each column of KB memory may be selected
            ht_temp = hidden[0].unsqueeze(1).unsqueeze(2).expand_as(embed_column)
            column_score = torch.sum(embed_column * ht_temp, dim=3) #b * m * s
            column_score = F.softmax(column_score, dim=2)  #b * m * s

            # query the KB memory using the hidden state of sketch RNN
            kb_prob_soft, kb_prob_logits = extKnow(query_vector, kb_global_pointer, False)
            kb_prob_logits_temp = kb_prob_logits.unsqueeze(2).expand_as(column_score)
            kb_prob_logits_temp = (1-1/column_arr.size(2)) * kb_prob_logits_temp + kb_prob_logits_temp * column_score

            for bi in range(batch_size):
                kb_prob_logits_temp[bi, kb_lengths[bi] - 1, 1:] = 0
                all_decoder_outputs_ptr_kb[t,bi] = kb_prob_logits_temp[bi].view(-1)


            if use_teacher_forcing:
                decoder_input = target_batches[:,t] 
            else:
                decoder_input = topvi.squeeze()
            
            if get_decoded_words:

                search_len_story = min(10, story_size[1])
                story_prob_soft = story_prob_soft * memory_mask_for_step_story
                story_topi, story_toppi = story_prob_soft.data.topk(search_len_story)

                search_len_kb = min(10, kb_size[1]*kb_size[2])
                kb_prob_logits_temp = kb_prob_logits_temp * memory_mask_for_step_kb
                kb_prob_soft_temp = self.softmax(kb_prob_logits_temp.view(batch_size,-1))
                kb_topi, kb_toppi = kb_prob_soft_temp.data.topk(search_len_kb)
                search_len = search_len_kb if search_len_kb < search_len_story else search_len_story

                temp_f, temp_c = [], []
                for bi in range(batch_size):
                    gate.append([])
                    probability_kb.append([])
                    probability_story.append([])
                    column_score_view.append([])
                    token = topvi[bi].item() #topvi[:,0][bi].item()
                    temp_c.append(self.lang.index2word[token])
                    if '@' in self.lang.index2word[token]:
                        column_score_view[bi].append([round(data,4) for data in column_score[bi,0,:].data.tolist()])
                        cw = 'UNK'
                        story = None
                        # choose the word selected in dialogue memory or the word selected in KB memory
                        for i in range(search_len):
                            if kb_toppi[:,i][bi] < (kb_lengths[bi]-1)*column_arr.size(2) and story_toppi[:,i][bi] < story_lengths[bi]-1:
                                if kb_topi[:,i][bi] > story_topi[:,i][bi]:
                                    gate[bi].append(1)
                                    probability_kb[bi].append(round(kb_topi[:,i][bi].item(), 4))
                                    probability_story[bi].append(round(story_topi[:, i][bi].item(), 4))
                                    cw = copy_list_kb[bi][kb_toppi[:, i][bi].item()]
                                    story = False
                                    break
                                else:
                                    gate[bi].append(0)
                                    probability_kb[bi].append(round(kb_topi[:, i][bi].item(), 4))
                                    probability_story[bi].append(round(story_topi[:, i][bi].item(), 4))
                                    cw = copy_list_story[bi][story_toppi[:, i][bi].item()]
                                    story = True
                                    break
                            elif story_toppi[:,i][bi] < story_lengths[bi]-1 and kb_toppi[:,i][bi] >= (kb_lengths[bi]-1) * column_arr.size(2):
                                gate[bi].append(0)
                                probability_kb[bi].append(round(kb_topi[:, i][bi].item(), 4))
                                probability_story[bi].append(round(story_topi[:, i][bi].item(), 4))
                                cw = copy_list_story[bi][story_toppi[:, i][bi].item()]
                                story = True
                                break
                            elif story_toppi[:,i][bi] >= story_lengths[bi]-1 and kb_toppi[:,i][bi] < (kb_lengths[bi]-1)*column_arr.size(2):
                                gate[bi].append(1)
                                probability_kb[bi].append(round(kb_topi[:, i][bi].item(), 4))
                                probability_story[bi].append(round(story_topi[:, i][bi].item(), 4))
                                cw = copy_list_kb[bi][kb_toppi[:, i][bi].item()]
                                story = False
                                break

                        temp_f.append(cw)
                        
                        if args['record']:
                            if story:
                                memory_mask_for_step_story[bi, story_toppi[:,i][bi].item()] = 0
                            else:
                                memory_mask_for_step_kb[bi, int(kb_toppi[:,i][bi].item()/column_arr.size(2)),kb_toppi[:,i][bi].item()%column_arr.size(2)] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)
        return all_decoder_outputs_vocab, all_decoder_outputs_ptr_story, all_decoder_outputs_ptr_kb, decoded_fine, decoded_coarse, column_score_view, gate, probability_kb, probability_story

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        # scores = F.softmax(scores_, dim=1)
        return scores_



class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
