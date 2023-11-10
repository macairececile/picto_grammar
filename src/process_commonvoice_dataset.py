import pandas as pd
import os

df = pd.read_csv('/data/macairec/PhD/Grammaire/corpus/csv/commonvoice/train_clean.csv')

slices = [df[i:i + 50000] for i in range(0, len(df), 50000)]

output_folder = '/data/macairec/PhD/Grammaire/corpus/csv/commonvoice/'
os.makedirs(output_folder, exist_ok=True)

for i, slice_df in enumerate(slices):
    output_filename = os.path.join(output_folder, f'train_commonvoice_{i + 1}.csv')
    slice_df.to_csv(output_filename, index=False)

print("Les fichiers ont été créés avec succès.")
