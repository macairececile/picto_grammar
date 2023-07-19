import json


def create_json(sentences_proc, s_rules, output_json):
    json_list = []

    for i, j in enumerate(sentences_proc):
        sentence = j
        s = s_rules[i]
        pictos = []
        if s != None:
            for w in s:
                if w.to_picto:
                    if w.picto != [404]:
                        pictos.append(str(w.picto[0]))
            f_pictos = ",".join(pictos)
            json_obj = {
                "sentence": sentence,
                "pictos": f_pictos
            }

            json_list.append(json_obj)
        else:
            print(i)

    json_str_1 = json.dumps(json_list[:100], indent=2, ensure_ascii=False)
    json_str_2 = json.dumps(json_list[100:200], indent=2, ensure_ascii=False)
    json_str_3 = json.dumps(json_list[200:300], indent=2, ensure_ascii=False)
    with open(output_json.split(".json")[0]+"_1.json", 'w', encoding='utf-8') as file:
        file.write(json_str_1)
    with open(output_json.split(".json")[0]+"_2.json", 'w', encoding='utf-8') as file:
        file.write(json_str_2)
    with open(output_json.split(".json")[0]+"_3.json", 'w', encoding='utf-8') as file:
        file.write(json_str_3)
