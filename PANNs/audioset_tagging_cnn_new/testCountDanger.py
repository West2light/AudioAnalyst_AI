import pandas as pd

df = pd.read_csv("audio_embeddings.csv")
print(df['label'].value_counts())