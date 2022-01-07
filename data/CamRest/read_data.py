import json
from nltk import wordpunct_tokenize as tokenizer
import argparse

def entity_replace(temp, bot, user, names={}):
    # change poi, location, event first
    global_rp = {
        "restaurants": "restaurant",
        "moderately": "moderate"
    }


def cleaner(token_array):
    new_token_array = []
    for idx, token in enumerate(token_array):
        temp = token
        if token == ".." or token == "." or token == "...": continue
        if (token == "am" or token == "pm") and token_array[idx - 1].isdigit():
            new_token_array.pop()
            new_token_array.append(token_array[idx - 1] + token)
            continue
        if token == ":),": continue
        if token == "avenue": temp = "ave"
        if token == "heavey" and "traffic" in token_array[idx + 1]: temp = "heavy"
        if token == "heave" and "traffic" in token_array[idx + 1]: temp = "heavy"
        if token == "'": continue
        if token == "-" and "0" in token_array[idx - 1]:
            new_token_array.pop()
            new_token_array.append(token_array[idx - 1] + "f")
            if "f" not in token_array[idx + 1]:
                token_array[idx + 1] = token_array[idx + 1] + "f"
        new_token_array.append(temp)
    return new_token_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--json', dest='json',
                        default='CamRest676_dev.json',
                        help='process json file')
    args = parser.parse_args()
    gen_file = open('CamRest676_dev.txt', 'w')


    with open(args.json) as f:
        dialogues = json.load(f)

    with open('CamRest_entities.json') as f:
        entities_dict = json.load(f)

    global_kb_type = ['name', 'area', 'food', 'phone', 'pricerange', "location",
                      'address', 'type', 'id', 'postcode']
    global_temp = []
    di = {}
    for e in global_kb_type:
        for p in map(lambda x: str(x).lower(), entities_dict[e]):
            di[p] = p
    global_temp.append(di)

    for d in dialogues:
        for el in d['scenario']['kb']['items']:
            slots = ['name', 'area', 'food', 'phone', 'pricerange', "location", 'address', 'type', 'id', 'postcode']
            # for slot in slots:
            #     el[slot] = " ".join(tokenizer(el[slot].replace("'"," "))).lower()
            di = {
                el['name']: el['name'],
                el['area']: el['area'],
                el['food']: el['food'],
                el['phone']: el['phone'],
                el['pricerange']: el['pricerange'],
                el['location']: el['location'],
                el['address']: el['address'],
                el['type']: el['type'],
                el['id']: el['id'],
                el['postcode']: el['postcode']

            }
            gen_file.write("0 "+" "+di[el['name']]+" "+di[el['area']]+" "+di[el['food']]+" "+di[el['phone']]+" "+di[el['pricerange']]+" "+di[el['location']]+" "+di[el['address']]+" "+di[el['type']]+" "+di[el['id']]+" "+di[el['postcode']])
            gen_file.write('\n')

        if(len(d['dialogue'])%2 != 0):
            d['dialogue'].pop()

        j = 1
        for i in range(0,len(d['dialogue']),2):
            user = " ".join(cleaner(tokenizer(str(d['dialogue'][i]['data']['utterance']).lower())))
            bot = " ".join(cleaner(tokenizer(str(d['dialogue'][i+1]['data']['utterance']).lower())))
            user = user.replace("restaurants", "restaurant")
            bot = bot.replace("restaurants", "restaurant")
            user = user.replace("moderately", "moderate")
            bot = bot.replace("moderately", "moderate")
            gold_entity = []
            for key in bot.split(' '):
                for e in global_kb_type:
                    for p in map(lambda x: str(x).lower(), entities_dict[e]):
                        if(key == p):
                            gold_entity.append(key)
                        elif(key == str(p).replace(" ", "_")):
                            gold_entity.append(key)
            gold_entity = list(set(gold_entity))
            if bot!="" and user!="":
                gen_file.write(str(j)+" "+user+'\t'+bot+'\t'+str(gold_entity))
                gen_file.write('\n')
                j+=1
        gen_file.write('\n')
    gen_file.close()


