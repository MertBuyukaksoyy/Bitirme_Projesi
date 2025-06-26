import pandas as pd
import os

def get_composer_mapping(metadata_path, max_pieces=15):
    df = pd.read_csv(metadata_path)

    # Sadece composer ve id (dosya numarası) gerekli
    df = df.dropna(subset=['composer']).drop_duplicates(subset=['id'])

    selected_df = df.iloc[:max_pieces]

    # WAV dosya adları (örneğin: 2382.wav)
    file_mapping = {row['id']: f"{row['id']}.wav" for _, row in selected_df.iterrows()}
    composer_mapping = dict(zip(selected_df['id'], selected_df['composer']))

    return composer_mapping, file_mapping
