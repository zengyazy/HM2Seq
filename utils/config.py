import os
import argparse
from tqdm import tqdm

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0 

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False
MAX_LENGTH = 10

parser = argparse.ArgumentParser(description='Seq_TO_Seq Dialogue bAbI')
parser.add_argument('-ds','--dataset', help='dataset, babi or kvr', required=False)
parser.add_argument('-hdd','--hidden', help='Hidden size', required=False)
parser.add_argument('-bsz','--batch', help='Batch_size', required=False)
parser.add_argument('-lr','--learn', help='Learning Rate', required=False)
parser.add_argument('-dr','--drop', help='Drop Out', required=False)
parser.add_argument('-um','--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)
parser.add_argument('-l','--layer', help='Layer Number', required=False)
parser.add_argument('-lm','--limit', help='Word Limit', required=False,default=-10000)
parser.add_argument('-path','--path', help='path of the file to load', required=False)
parser.add_argument('-clip','--clip', help='gradient clipping', required=False, default=10)
parser.add_argument('-tfr','--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False, default=0.5)

parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-an','--addName', help='An add name for the save folder', required=False, default='')
parser.add_argument('-gs','--genSample', help='Generate Sample', required=False, default=0)
parser.add_argument('-es','--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')
parser.add_argument('-rec','--record', help='use record function during inference', type=int, required=False, default=0)

parser.add_argument('-f','--flag', help='MEM_TOKEN_SIZE flag', required=False, default=0)

args = vars(parser.parse_args())
print(str(args))
print("USE_CUDA: "+str(USE_CUDA))

LIMIT = int(args["limit"]) 
if args["dataset"] == 'kvr_navigate' or args["dataset"] == 'kvr_weather':
    MEM_TOKEN_SIZE = 5
elif args["dataset"] == 'kvr_schedule':
    MEM_TOKEN_SIZE = 6
elif args["dataset"] == 'camrest':
    MEM_TOKEN_SIZE = 10