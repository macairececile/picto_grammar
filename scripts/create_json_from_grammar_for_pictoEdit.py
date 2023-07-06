import json

def create_json(dataframe, s_rules, output_json):
    json_list = []

    for i, row in dataframe.iterrows():
        sentence = row['text']
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

    json_str = json.dumps(json_list, indent=2, ensure_ascii=False)
    with open(output_json, 'w', encoding='utf-8') as file:
        file.write(json_str)