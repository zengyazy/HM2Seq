import json
from nltk import wordpunct_tokenize as tokenizer
import argparse

def entity_replace(temp, bot, user, names={}):   
    # change poi, location, event first
    global_rp = {
        "pf changs": "p_._f_._changs",
        "p f changs": "p_._f_._changs",
        "'": "",
        "restaurants": "restaurant",
        "activities": "activity",
        "appointments": "appointment",
        "doctors": "doctor",
        "doctor s": "doctor",
        "optometrist s": "optometrist",
        "conferences": "conference",
        "meetings": "meeting",
        "labs": "lab",
        "stores": "store",
        "stops": "stop",
        "centers": "center",
        "garages": "garage",
        "stations": "station",
        "hospitals": "hospital"
    }
    
    for grp in global_rp.keys():
        bot = bot.replace(grp, global_rp[grp])
        user = user.replace(grp, global_rp[grp])

    for name in names.keys():  
        if name in bot:
            bot = bot.replace(name, names[name])
        if name in user:
            user = user.replace(name, names[name]) 

    for e in temp:
        for wo in e.keys():
            inde = bot.find(wo)
            if(inde!=-1):
                bot = bot.replace(wo, e[wo])
            inde = user.find(wo)
            if(inde!=-1):
                user = user.replace(wo, e[wo]) 
    return bot, user

def cleaner(token_array):
    new_token_array = []
    for idx, token in enumerate(token_array):
        temp = token
        if token==".." or token=="." or token=="...": continue
        if (token=="am" or token=="pm") and token_array[idx-1].isdigit():
            new_token_array.pop()
            new_token_array.append(token_array[idx-1]+token)
            continue
        if token==":),": continue
        if token=="avenue": temp = "ave"
        if token=="heavey" and "traffic" in token_array[idx+1]: temp = "heavy"
        if token=="heave" and "traffic" in token_array[idx+1]: temp = "heavy"
        if token=="'": continue
        if token=="-" and "0" in token_array[idx-1]: 
            new_token_array.pop()
            new_token_array.append(token_array[idx-1]+"f")
            if "f" not in token_array[idx+1]:
                token_array[idx+1] = token_array[idx+1]+"f"
        new_token_array.append(temp)
    return new_token_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--json', dest='json',
                        default='kvret_dev_public.json',
                        help='process json file')
    args = parser.parse_args()
    gen_file_weather = 'weather_dev.txt'
    gen_weather = open(gen_file_weather,"w")

    gen_file_navigate = 'navigate_dev.txt'
    gen_navigate = open(gen_file_navigate, "w")

    gen_file_schedule = 'schedule_dev.txt'
    gen_schedule = open(gen_file_schedule, "w")

    with open(args.json) as f:
        dialogues = json.load(f)

    with open('kvret_entities.json') as f:
        entities_dict = json.load(f)

    global_kb_type = ['distance','traffic_info','location', 'weather_attribute','temperature',"weekly_time", 'event', 'time','date','party','room','agenda']
    global_temp = []
    di = {}
    for e in global_kb_type:
        for p in map(lambda x: str(x).lower(), entities_dict[e]):       
            if "_" in p and p.replace("_"," ") != p:
                di[p.replace("_"," ")] = p 
            else:
                if p != p.replace(" ","_"):
                    di[p] = p.replace(" ","_")
    global_temp.append(di)

    for d in dialogues:
        if(d['scenario']['task']['intent']=="navigate"): #"schedule" "navigate"
            # print("#navigate#")
            gen_navigate.write("#navigate#")
            gen_navigate.write('\n')
            temp = []
            names = {}
            for el in d['scenario']['kb']['items']:
                poi = " ".join(tokenizer(el['poi'].replace("'"," "))).replace(" ", "_").lower()
                slots = ['poi','distance','traffic_info','poi_type','address']
                for slot in slots:
                    el[slot] = " ".join(tokenizer(el[slot].replace("'"," "))).lower()
                names[el['poi']] = poi
                di = {
                    el['distance']: el['distance'].replace(" ", "_"),
                    el['traffic_info']: el['traffic_info'].replace(" ", "_"),
                    el['poi_type']: el['poi_type'].replace(" ", "_"),
                    el['address']: el['address'].replace(" ", "_"),
                }
                # print(di)
                gen_navigate.write("0 "+poi+" "+di[el['poi_type']]+" "+di[el['distance']]+" "+di[el['traffic_info']]+" "+di[el['address']])
                gen_navigate.write('\n')
                temp.append(di)
            temp += global_temp

            if(len(d['dialogue'])%2 != 0):
                d['dialogue'].pop()
            
            j = 1
            for i in range(0,len(d['dialogue']),2):
                user = " ".join(cleaner(tokenizer(str(d['dialogue'][i]['data']['utterance']).lower())))
                bot = " ".join(cleaner(tokenizer(str(d['dialogue'][i+1]['data']['utterance']).lower())))
                bot, user = entity_replace(temp, bot, user, names) 
                navigation = global_kb_type #['distance','traffic_info']
                nav_poi = ['address','poi','type']
                gold_entity = []
                for key in bot.split(' '):                    
                    for e in navigation:
                        for p in map(lambda x: str(x).lower(), entities_dict[e]):      
                            if(key == p):
                                gold_entity.append(key)  
                            elif(key == str(p).replace(" ", "_")):
                                gold_entity.append(key) 
                    
                    for e in entities_dict['poi']:
                        for p in nav_poi:
                            if(key == str(e[p]).lower()):
                                gold_entity.append(key)  
                            elif(key == str(e[p]).lower().replace(" ", "_")):
                                gold_entity.append(key)
                gold_entity = list(set(gold_entity))
                if bot!="" and user!="":
                    gen_navigate.write(str(j)+" "+user+'\t'+bot+'\t'+str(gold_entity))
                    gen_navigate.write('\n')
                    j+=1
            gen_navigate.write('\n')

        elif (d['scenario']['task']['intent']=="weather"): #"schedule" "navigate"
            # print("#weather#")
            gen_weather.write("#weather#")
            gen_weather.write('\n')
            temp = []
            j = 1
            for el in d['scenario']['kb']['items']:
                
                for el_key in el.keys():
                    el[el_key] = " ".join(tokenizer(el[el_key])).lower()
                loc = el['location'].replace(" ", "_")
                di = {el['location']: loc}
                temp.append(di)
                days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
                for day in days:
                    gen_weather.write("0 " +loc+" "+day+" "+el[day].split(',')[0].rstrip().replace(" ", "_")+" "+el[day].split(',')[1].split(" ")[3]+" "+el[day].split(',')[2].split(" ")[3])
                    gen_weather.write('\n')
            temp += global_temp
            gen_weather.write("0 today " + d['scenario']['kb']['items'][0]["today"])
            gen_weather.write('\n')

            if(len(d['dialogue'])%2 != 0):
                d['dialogue'].pop()
            
            for i in range(0,len(d['dialogue']),2):
                user = " ".join(cleaner(tokenizer(str(d['dialogue'][i]['data']['utterance']).lower())))
                bot = " ".join(cleaner(tokenizer(str(d['dialogue'][i+1]['data']['utterance']).lower())))
                bot, user = entity_replace(temp, bot, user) 
                weather = global_kb_type #['location', 'weather_attribute','temperature',"weekly_time"]
                gold_entity = []
                for key in bot.split(' '):                    
                    for e in weather:
                        for p in map(lambda x: str(x).lower(), entities_dict[e]):      
                            if(key == p):
                                gold_entity.append(key)  
                            elif(key == str(p).replace(" ", "_")):
                                gold_entity.append(key) 
                gold_entity = list(set(gold_entity))
                if bot!="" and user!="":
                    gen_weather.write(str(j)+" "+user+'\t'+bot+'\t'+str(gold_entity))
                    gen_weather.write('\n')
                    j+=1  
            # print("")
            # gen_f.write("")
            gen_weather.write('\n')

        if(d['scenario']['task']['intent']=="schedule"): #"schedule"
            # print("#schedule#")
            gen_schedule.write("#schedule#")
            gen_schedule.write('\n')
            temp = []
            names = {}
            j=1
            if(d['scenario']['kb']['items'] != None):
                for el in d['scenario']['kb']['items']:
                    for el_key in el.keys():
                        el[el_key] = " ".join(tokenizer(el[el_key])).lower()
                    ev = el['event'].replace(" ", "_")
                    names[el['event']] = ev
                    slots = ['time','date','party','room','agenda']
                    di = {}
                    for slot in slots:
                        if el[slot]=="-":
                            el[slot] = "PAD"
                            continue
                        if slot == "time":
                            di[el[slot]] = el[slot].replace(" ", "")
                        else:
                            di[el[slot]] = el[slot].replace(" ", "_")
                    temp.append(di)
                    gen_schedule.write("0 "+ev+" "+el['date'].replace(" ", "_")+" "+el['time'].replace(" ", "")+" "+el['party'].replace(" ", "_")+" "+el['room'].replace(" ", "_")+" "+el['agenda'].replace(" ", "_"))
                    gen_schedule.write('\n')
            temp += global_temp

            if(len(d['dialogue'])%2 != 0):
                d['dialogue'].pop()

            for i in range(0,len(d['dialogue']),2):
                user = " ".join(cleaner(tokenizer(str(d['dialogue'][i]['data']['utterance']).lower())))
                bot = " ".join(cleaner(tokenizer(str(d['dialogue'][i+1]['data']['utterance']).lower())))         
                bot, user = entity_replace(temp, bot, user, names)  
                calendar = global_kb_type #['event','time', 'date', 'party', 'room', 'agenda']
                gold_entity = []
                for key in bot.split(' '):                    
                    for e in calendar:
                        for p in map(lambda x: str(x).lower(), entities_dict[e]):      
                            if(key == p):
                                gold_entity.append(key)  
                            elif(key == str(p).replace(" ", "_")):
                                gold_entity.append(key) 
                gold_entity = list(set(gold_entity))
                if bot!="" and user!="":
                    gen_schedule.write(str(j)+" "+user+'\t'+bot+'\t'+str(gold_entity))
                    gen_schedule.write('\n')
                    j+=1
            gen_schedule.write("\n")
    gen_weather.close()
    gen_navigate.close()
    gen_schedule.close()
