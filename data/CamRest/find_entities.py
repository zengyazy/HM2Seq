import json

file_in_train = "CamRest676_train.json"
file_in_dev = "CamRest676_dev.json"
file_in_test = "CamRest676_test.json"
gen_file = "CamRest_entities.json"

global_kb_type = ['name', 'area', 'food', 'phone', 'pricerange', "location", 'address', 'type', 'id', 'postcode']
global_entities = {}
for kb_type in global_kb_type:
    global_entities[kb_type] = []

def read_file(file):
    with open(file) as f:
        dialogues = json.load(f)
        for dialogue in dialogues:
            for el in dialogue['scenario']['kb']['items']:
                for kb_type in global_kb_type:
                    if el[kb_type] not in global_entities[kb_type]:
                        global_entities[kb_type].append(el[kb_type])

if __name__ == "__main__":
    read_file(file_in_train)
    read_file(file_in_dev)
    read_file(file_in_test)
    with open(gen_file,'w') as gen_file:
        json.dump(global_entities, gen_file, indent=1)

