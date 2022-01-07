import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
import json
import sys
import matplotlib.pyplot as plt

from utils.measures import wer, moses_multi_bleu
from utils.masked_cross_entropy import *
from utils.config import *
from models.modules import *


class HM2Seq(nn.Module):
    def __init__(self, hidden_size, lang, max_resp_len, path, lr, n_layers, dropout):
        super(HM2Seq, self).__init__()
        self.name = "HM2Seq"
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size    
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.decoder_hop = n_layers
        self.softmax = nn.Softmax(dim=0)

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.extKnow = torch.load(str(path)+'/enc_kb.th')
                self.decoder = torch.load(str(path)+'/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.extKnow = torch.load(str(path)+'/enc_kb.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
        else:
            self.encoder = ContextRNN(lang.n_words, hidden_size, dropout)
            self.extKnow = ExternalKnowledge(lang.n_words, hidden_size, n_layers, dropout)
            self.decoder = LocalMemoryDecoder(self.encoder.embedding, lang, hidden_size, self.decoder_hop, dropout) #Generator(lang, hidden_size, dropout)

        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.extKnow_optimizer = optim.Adam(self.extKnow.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        self.criterion_bce = nn.BCELoss()
        self.reset()

        if USE_CUDA:
            self.encoder.cuda()
            self.extKnow.cuda()
            self.decoder.cuda()

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_gs = self.loss_gs / self.print_every
        print_loss_gk = self.loss_gk / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_ls = self.loss_ls / self.print_every
        print_loss_lk = self.loss_lk / self.print_every
        self.print_every += 1     
        return 'L:{:.2f},LES:{:.2f},LEK:{:.2f},LG:{:.2f},LPS:{:.2f},LPK:{:.2f}'.format(print_loss_avg, print_loss_gs, print_loss_gk, print_loss_v, print_loss_ls, print_loss_lk)
    
    def save_model(self, dec_type):
        name_data = args['dataset'] + '/'
        layer_info = str(self.n_layers)
        directory = 'save/'+args["addName"]+name_data+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+'L'+layer_info+'lr'+str(self.lr)+str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.extKnow, directory + '/enc_kb.th')
        torch.save(self.decoder, directory + '/dec.th')

    def reset(self):
        self.loss, self.print_every, self.loss_gs, self.loss_gk, self.loss_v, self.loss_ls, self.loss_lk = 0, 1, 0, 0, 0, 0, 0
    
    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.extKnow_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        # Encode and Decode
        use_teacher_forcing = random.random() < args['teacher_forcing_ratio'] 
        max_target_length = max(data['response_lengths'])
        all_decoder_outputs_vocab, all_decoder_outputs_ptr_story, all_decoder_outputs_ptr_kb, _, _, global_pointer_story, global_pointer_kb, _, _, _, _ = self.encode_and_decode(data, max_target_length, use_teacher_forcing, False)
        
        # Loss calculation and backpropagation
        loss_gs = self.criterion_bce(global_pointer_story, data['selector_index_story'])
        loss_gk = self.criterion_bce(global_pointer_kb, data['selector_index_kb'])
        loss_v = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), 
            data['sketch_response'].contiguous(), 
            data['response_lengths'])
        loss_ls = masked_cross_entropy(
            all_decoder_outputs_ptr_story.transpose(0, 1).contiguous(),
            data['ptr_index_story'].contiguous(),
            data['response_lengths'])
        loss_lk = masked_cross_entropy(
            all_decoder_outputs_ptr_kb.transpose(0, 1).contiguous(),
            data['ptr_index_kb'].contiguous(),
            data['response_lengths'])
        loss = loss_gs + loss_gk + loss_v + loss_ls + loss_lk
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        ec = torch.nn.utils.clip_grad_norm_(self.extKnow.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.extKnow_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_gs += loss_gs.item()
        self.loss_gk += loss_gk.item()
        self.loss_v += loss_v.item()
        self.loss_ls += loss_ls.item()
        self.loss_lk += loss_lk.item()

    def encode_and_decode(self, data, max_target_length, use_teacher_forcing, get_decoded_words):
        # Build unknown mask for memory
        if args['unk_mask'] and self.decoder.training:
            story_size = data['context_arr'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))], 1-self.dropout)[0]
            rand_mask[:,:,0] = rand_mask[:,:,0] * bi_mask
            conv_rand_mask = np.ones(data['conv_arr'].size())
            for bi in range(story_size[0]):
                end = data['conv_arr_lengths'][bi]
                conv_rand_mask[:end,bi,:] = rand_mask[bi,:end,:]
            rand_mask = self._cuda(rand_mask)
            conv_rand_mask = self._cuda(conv_rand_mask)
            conv_story = data['conv_arr'] * conv_rand_mask.long()
            story = data['context_arr'] * rand_mask.long()
            kb = data['kb_arr']
        else:
            story, conv_story = data['context_arr'], data['conv_arr']
            kb = data['kb_arr']
        kb_column = data['column_arr']
        
        # Encode dialog history and KB to vectors
        dh_outputs, dh_hidden = self.encoder(conv_story, data['conv_arr_lengths'])
        global_pointer_story, global_pointer_kb, kb_readout = self.extKnow.load_memory(story, kb, kb_column, data['kb_arr_lengths'], data['conv_arr_lengths'], dh_hidden, dh_outputs)
        encoded_hidden = torch.cat((dh_hidden, kb_readout), dim=1)
        
        # Get the words that can be copy from the memory
        batch_size = len(data['context_arr_lengths'])
        self.copy_list_story = []
        for elm in data['context_arr_plain']:
            elm_temp = [ word_arr[0] for word_arr in elm ]
            self.copy_list_story.append(elm_temp)

        self.copy_list_kb = []
        for elm in data['kb_arr_plain']:
            elm_temp = [word for word_arr in elm for word in word_arr]
            self.copy_list_kb.append(elm_temp)
        
        outputs_vocab, outputs_ptr_story, outputs_ptr_kb, decoded_fine, decoded_coarse, column_score_view, gate, probability_kb, probability_story = self.decoder.forward(
            self.extKnow,
            data['column_arr'],
            story.size(),
            kb.size(),
            data['context_arr_lengths'],
            data['kb_arr_lengths'],
            self.copy_list_story,
            self.copy_list_kb,
            encoded_hidden, 
            data['sketch_response'], 
            max_target_length, 
            batch_size, 
            use_teacher_forcing, 
            get_decoded_words, 
            global_pointer_story,
            global_pointer_kb)

        return outputs_vocab, outputs_ptr_story, outputs_ptr_kb, decoded_fine, decoded_coarse, global_pointer_story, global_pointer_kb, column_score_view, gate, probability_kb, probability_story

    def evaluate(self, dev, matric_best, early_stop=None):
        print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.extKnow.train(False)
        self.decoder.train(False)

        ref, hyp = [], []
        acc, total = 0, 0
        dialog_acc_dict = {}
        F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
        F1_count, F1_cal_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
        pbar = tqdm(enumerate(dev),total=len(dev), ascii=True)
        new_precision, new_recall, new_f1_score = 0, 0, 0

        if args['dataset'].split('_')[0] == 'kvr':
            TP_all, FP_all, FN_all = 0, 0, 0
            with open('data/KVR/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))
        elif args['dataset'] == 'camrest':
            with open('data/CamRest/CamRest_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    global_entity_list += global_entity[key]
                global_entity_list = list(set(global_entity_list))

        for j, data_dev in pbar:
            # Encode and Decode
            _, _, _, decoded_fine, decoded_coarse, _, _, _, _,_,_ = self.encode_and_decode(data_dev, self.max_resp_len, False, True)
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS': break
                    else: st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS': break
                    else: st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)

                if args['dataset'].split('_')[0] == 'kvr':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                elif args['dataset'] == 'camrest':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                else:
                    # compute Dialogue Accuracy Score
                    current_id = data_dev['ID'][bi]
                    if current_id not in dialog_acc_dict.keys():
                        dialog_acc_dict[current_id] = []
                    if gold_sent == pred_sent:
                        dialog_acc_dict[current_id].append(1)
                    else:
                        dialog_acc_dict[current_id].append(0)

                # compute Per-response Accuracy Score
                total += 1
                if (gold_sent == pred_sent):
                    acc += 1

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)

        # Set back to training mode
        self.encoder.train(True)
        self.extKnow.train(True)
        self.decoder.train(True)
        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        acc_score = acc / float(total)
        print("ACC SCORE:\t"+str(acc_score))

        if args['dataset'].split('_')[0] == 'kvr':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{}".format(F1_pred/float(F1_count)))
            print("BLEU SCORE:\t"+str(bleu_score))
        elif args['dataset'] == 'camrest':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{}".format(F1_pred / float(F1_count)))
            print("BLEU SCORE:\t" + str(bleu_score))
        else:
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k])==sum(dialog_acc_dict[k]):
                    dia_acc += 1
            print("Dialog Accuracy:\t"+str(dia_acc*1.0/len(dialog_acc_dict.keys())))

        if (early_stop == 'BLEU'):
            if (bleu_score >= matric_best):
                self.save_model('BLEU-'+str(bleu_score))
                print("MODEL SAVED")
            return bleu_score
        elif (early_stop == 'ENTF1'):
            if (F1_score >= matric_best or F1_score > 0.48):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")
            return F1_score
        else:
            if (acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(acc_score))
                print("MODEL SAVED")
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count

    def compute_F1(self, precision, recall):
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        return F1


    def print_examples(self, batch_idx, data, pred_sent, pred_sent_coarse, gold_sent):
        kb_len = len(data['context_arr_plain'][batch_idx])-data['conv_arr_lengths'][batch_idx]-1
        print("{}: ID{} id{} ".format(data['domain'][batch_idx], data['ID'][batch_idx], data['id'][batch_idx]))
        for i in range(kb_len):
            kb_temp = [w for w in data['context_arr_plain'][batch_idx][i] if w!='PAD']
            kb_temp = kb_temp[::-1]
            if 'poi' not in kb_temp:
                print(kb_temp)
        flag_uttr, uttr = '$u', []
        for word_idx, word_arr in enumerate(data['context_arr_plain'][batch_idx][kb_len:]):
            if word_arr[1]==flag_uttr:
                uttr.append(word_arr[0])
            else:
                print(flag_uttr,': ', " ".join(uttr))
                flag_uttr = word_arr[1]
                uttr = [word_arr[0]]
        print('Sketch System Response : ', pred_sent_coarse)
        print('Final System Response : ', pred_sent)
        print('Gold System Response : ', gold_sent)
        print('\n')
