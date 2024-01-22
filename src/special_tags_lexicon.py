import os
import json
import csv

# Spécifiez le chemin du dossier contenant les fichiers JSON
dossier_json = '/data/macairec/Cloud/PROPICTO_RESSOURCES/ARASAAC/ARASAAC_Pictos_all_json'


# Fonction pour lire et extraire les informations nécessaires d'un fichier JSON
def lire_json(fichier_json):
    with open(fichier_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        _id = data.get("_id", None)
        sex = data.get("sex", None)
        violence = data.get("violence", None)
        return _id, sex, violence


# Liste pour stocker les informations extraites de tous les fichiers JSON
informations = []

# Parcourir tous les fichiers dans le dossier
for fichier in os.listdir(dossier_json):
    if fichier.endswith('.json'):
        chemin_fichier = os.path.join(dossier_json, fichier)
        _id, sex, violence = lire_json(chemin_fichier)
        informations.append({"id_picto": _id, "sex": sex, "violence": violence})

# Spécifiez le chemin du fichier CSV de sortie
fichier_csv = '/data/macairec/PhD/Grammaire/dico/tags.csv'

# Écrire les informations dans le fichier CSV
with open(fichier_csv, 'w', newline='', encoding='utf-8') as csv_file:
    colonnes = ["id_picto", "sex", "violence"]
    writer = csv.DictWriter(csv_file, fieldnames=colonnes)

    # Écrire l'en-tête du fichier CSV
    writer.writeheader()

    # Écrire les informations extraites dans le fichier CSV
    writer.writerows(informations)

print(f"Les informations ont été enregistrées dans {fichier_csv}.")
