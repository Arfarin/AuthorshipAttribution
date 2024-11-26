import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re

# CSV-Datei mit Erklärungen einlesen
datei_pfad = './Results11Authors/Results_explanations.csv'
df = pd.read_csv(datei_pfad, delimiter=';')

# Deutsche Stopwörter laden
nltk.download('stopwords')
stopwords_german = set(stopwords.words('german'))

# Eigene zusätzliche Stopwörter hinzufügen
unbedeutendeWörter = {'hin.', 'typisch', 'Textausschnitt', 'Merkmale', 'Hinweise', 'Text', 
                      'Erklärung:', 'ist.', 'deuten', 'stark', 'insbesondere', 
                      'Autor', 'Autoren', 'kommen', 'Zuordnung', 'Hinweise:', 
                      'Bruce' 'Coville','Astrid', 'Lindgren', "Lindgrens", 'Jeanne', 'Birdsall', 'Cornelia', 'Funke', 'Funkes','Antoine' ,'de', 'Saint-Exupéry','Sarah', 'Lark', 'Larks',
                        'Jonathan', 'Swift',
                        'Robert', 'Louis', 'Stevenson', 'Stevensons',
                        'Sir', 'Arthur', 'Conan', 'Doyle', 'Doyles'
                        'Horace' ,'Walpole',
                        'Oscar', 'Wilde',
                        'Bram', 'Stoker',
                        'Mary','Shelley',
                        'Charlotte', 'Brontë',
                        'Gaston', 'Leroux',
                        'Emily', 'Brontë', 'Brontës',
                        'William' 'Russell',
                        'Michael', 'Ende',
                        'Jane' ,'Austen', 'Austens',
                        'Tolkien', 'Tolkiens',
                        'Susanne','Abel', "Themen", "Fazit:"
}

# Funktion zur Verarbeitung und zum Vergleich
def remove_stopwords_and_compare_top_words(df, stopwords):
    """
    Entfernt Stopwörter aus den Spalten eines DataFrames, zählt Wortfrequenzen 
    und gibt die häufigsten 5 Wörter jeder Spalte aus.
    
    Args:
        df: Das Pandas DataFrame mit den Textspalten.
        stopwords: Eine Menge von Stopwörtern.
    """
    # Konvertiere alle Spalten zu Strings
    df = df.astype(str)

    # Wörterbuch, um Top-5-Wörter jeder Spalte zu speichern
    column_top_words = {}

    for column in df.columns:
        # Text aus der Spalte in Wörter zerlegen
        words = []
        for text in df[column]:
            words.extend(text.split())

        # Entferne Stopwörter
        words = [word for word in words if word.lower() not in stopwords and word not in unbedeutendeWörter]

        # Wortfrequenzen zählen
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Top 5 Wörter finden
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        column_top_words[column] = [word for word, _ in top_words]  # Nur Wörter speichern
        print(f"Top 5 Wörter in Spalte {column} nach Entfernung von Stopwörtern:")
        for word, freq in top_words:
            print(f"{word}: {freq}")
        print()
        
         # Wordcloud erstellen und speichern
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Entfernt weißen Rand
        column = "wordcloud_" + column
        plt.savefig(f"{column}.png", bbox_inches='tight', pad_inches=0)  # Speichern ohne Rand
        plt.close()

    # Vergleich der häufigsten Wörter
    print("Vergleich der häufigsten Wörter:")
    for column, top_words in column_top_words.items():
        print(f"Spalte {column}: {top_words}")

    # Gemeinsame Wörter in den Top-5-Listen
    all_top_words = [word for top_words in column_top_words.values() for word in top_words]
    common_words = {word for word in all_top_words if all_top_words.count(word) > 1}

    print("\nWörter, die in mehr als einer Spalte unter den häufigsten vorkommen:")
    if common_words:
        print(", ".join(common_words))
    else:
        print("Keine gemeinsamen Wörter gefunden.")

# Funktion ausführen
remove_stopwords_and_compare_top_words(df, stopwords_german | unbedeutendeWörter)
