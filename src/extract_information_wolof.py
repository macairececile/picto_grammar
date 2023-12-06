import xml.etree.ElementTree as ET

# Charger le fichier XML
tree = ET.parse("votre_fichier.xml")
root = tree.getroot()

# Parcourir chaque élément SYNSET
for synset in root.findall('.//SYNSET'):
    # Extraire le contenu entre les balises LITERAL
    literal = synset.find('./SYNONYM/LITERAL').text
    
    # Extraire le contenu de la balise ID
    synset_id = synset.find('./ID').text
    
    # Afficher les informations extraites
    print(f"ID: {synset_id}, LITERAL: {literal}")

