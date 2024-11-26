import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
import os
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader




# CSV-Datei mit Explanations einlesen
datei_pfad = './Data_11Authors/test_mitLabel.csv'
df = pd.read_csv(datei_pfad, delimiter=',')  
print(df.head)

# Extrahiere die Textausschnitte und Autoren-IDs
texts = df['Textausschnitt'].tolist()
author_ids = df['Author_id'].tolist()

# Konvertiere die Autoren-IDs in numerische Labels für die Klassifikation
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
author_labels = label_encoder.fit_transform(author_ids)  # Konvertiert Autoren-IDs in numerische Werte


## Model

# Lade das Tokenizer und Modell von Hugging Face
model_name = "gpt2"  # Du kannst auch ein anderes Modell verwenden
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=len(set(author_labels)))

# GPT-2 benötigt eine Anpassung für Klassifikationsaufgaben. Hier fügen wir eine Klassifikationsschicht hinzu.
# GPT-2 hat normalerweise keine Padding-Token-Id, also fügen wir sie hinzu:
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id




# Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Erstelle das Dataset
dataset = TextDataset(texts, author_labels, tokenizer)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)


# Vorhersage

# Wähle den zweiten Textausschnitt aus
text_sample = df['Textausschnitt'].iloc[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoding = tokenizer(text_sample, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)

# Mache eine Vorhersage
model.eval()
with torch.no_grad():
    outputs = model(**encoding)
    predictions = torch.argmax(outputs.logits, dim=-1).item()

# Gib den vorhergesagten Autor aus
predicted_author = label_encoder.inverse_transform([predictions])[0]
print("Vorhergesagter Autor:", predicted_author)