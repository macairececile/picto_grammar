"""
Script to generate the json file for the PostEdit platform to post-edit sentences.
"""

import json


def create_json(sentences_proc, s_rules, output_json):
    """
        Create the json file with sentence and picto informations.

        Arguments
        ---------
        sentences_proc: str
        s_rules: list
        output_json: str
    """
    json_list = []

    for i, j in enumerate(sentences_proc):
        sentence = j
        s = s_rules[i]
        pictos = []
        if s is not None:
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

    json_str_1 = json.dumps(json_list, indent=2, ensure_ascii=False)
    with open(output_json, 'w', encoding='utf-8') as file:
        file.write(json_str_1)
