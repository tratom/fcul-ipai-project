import pandas as pd
from transformers import pipeline
from tqdm import tqdm

df_ntsb = pd.read_csv('data_sources/binding/ntsb_probable_cause.csv')
# Inizializza il classificatore zero-shot
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Definisci le nuove categorie
labels = [
    "Human Error – Control",
    "Human Error – Procedural",
    "Mechanical Failure",
    "Fuel Management",
    "Environmental Conditions",
    "Collision / Obstacle",
    "Loss of Situational Awareness",
    "Unknown / Not Determined"
]

# Funzione per classificare una singola frase
def classify_zero_shot(text):
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return "Unknown / Not Determined"
    result = classifier(text, candidate_labels=labels, multi_label=False)
    return result['labels'][0]  # etichetta con punteggio più alto

# Applica con progress bar
tqdm.pandas()
df_ntsb['ProbableCause_ZeroShot'] = df_ntsb['ProbableCause'].progress_apply(classify_zero_shot)


# Salva il DataFrame aggiornato in un nuovo file CSV
df_ntsb.to_csv("data_sources/binding/ntsb_with_zero_shot.csv", index=False)