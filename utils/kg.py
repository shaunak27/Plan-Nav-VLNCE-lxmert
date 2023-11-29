import json
from os import read

# TODO: read list of objects in matterport
# TODO: read list of objects in instructions

relationships = json.load(open('./relationships.json'))
cat2vocab = json.load(open('../data/models/cat2vocab.json'))
rooms = json.load(open('../data/models/rooms.json'))
total_images = len(relationships)
num_images = total_images
num_rels = 1000
rel_counter = {}

def add_relationship(sub_name, obj_name):
    """ Add object relationship to subject. """
    if rel_counter.get(sub_name) == None:
        rel_counter[sub_name] = {}
    
    if rel_counter[sub_name].get(obj_name) == None:
        rel_counter[sub_name][obj_name] = 1
    else:
        rel_counter[sub_name][obj_name] += 1

def post_process(rel_dict, min_occurrences=3):
    for sub in list(rel_dict.keys()):
        for obj in list(rel_dict[sub].keys()):
            if rel_dict[sub][obj] < min_occurrences:
                del rel_dict[sub][obj]
                
        if not rel_dict[sub]:
            del rel_dict[sub]

combined = list(cat2vocab.keys()) + list(cat2vocab.values()) +rooms

for i in range(total_images):
    if i > num_images:
        break
    
    rel = relationships[i]['relationships']
    for j in range(len(rel)):
        if j > num_rels:
            break
        
        obj, sub = rel[j]['object'], rel[j]['subject'] 
        if not obj.get('name') is None and not sub.get('name') is None:
            obj_name, sub_name = obj['name'], sub['name']
            if obj_name not in combined or sub_name not in combined:
                continue
            add_relationship(sub_name, obj_name)
        else:
            # ignoring this case
            if sub.get('name') == None:
                continue
            sub_name = sub['name']
            
            obj_names = obj['names'] if obj.get('name') is None else [obj['name']]
                
            for obj_name in obj_names:
                if obj_name not in combined or sub_name not in combined:
                    continue
                add_relationship(sub_name, obj_name)

post_process(rel_counter)

# print(f"Relationship counter:\n{json.dumps(rel_counter, indent=2)}")

with open('relationship_counter.json', 'w') as outfile:
    json.dump(rel_counter, outfile, indent=4, sort_keys=True)