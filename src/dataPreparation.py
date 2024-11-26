import pandas as pd
import numpy as np

# CSV-Datei einlesen
datei_pfad = 'LiteraturDatenMitTexten_prepared.csv'
df = pd.read_csv(datei_pfad, delimiter=';')

# Entfernen von Leerzeichen am Anfang und Ende der Autorennamen
df['Author'] = df['Author'].str.strip()

# Alle Datensätze mit Genre "Lehrbuch" entfernen
df = df[df['Genre'] != 'Lehrbuch']
# Alle Datensätze mit Sprache "en" entfernen
df = df[df['Sprache'] != 'en']

# Berechne die Häufigkeit jedes Autors
author_counts = df['Author'].value_counts()
# Filtere Autoren, die weniger als 9 Vorkommen haben
authors_to_keep = author_counts[author_counts >= 9].index
# Filtere das DataFrame, um nur die Zeilen mit diesen Autoren beizubehalten
filtered_df = df[df['Author'].isin(authors_to_keep)]

# FÜR REDUCED DATASET
# Wähle zufällig x Autoren aus und beschränke auf diese x Autoren
selected_authors = filtered_df['Author'].drop_duplicates().sample(8, random_state=1)
filtered_df = filtered_df[filtered_df['Author'].isin(selected_authors)]

anzahl_autoren = filtered_df['Author'].nunique()
print(f"Anzahl der verschiedenen Autoren: {anzahl_autoren}")

# Funktion, um genau 9 Zeilen pro Autor zu wählen und die Zeilenanzahl pro Titel gleichmäßig zu verteilen
def select_balanced_rows(df, author, n=9):
    # Filtere die Zeilen des aktuellen Autors
    author_df = df[df['Author'] == author]

    # Hole alle Titel für den aktuellen Autor
    titles = author_df['Titel'].unique()
    num_titles = len(titles)

    # Berechne die gleichmäßige Verteilung
    base_count = n // num_titles
    remainder = n % num_titles

    # Initialisiere eine Liste für die ausgewählten Zeilen
    selected_rows = []

    # Füge eine gleichmäßige Anzahl an Zeilen pro Titel hinzu
    for title in titles:
        # Anzahl der Zeilen für diesen Titel
        count = base_count + (1 if remainder > 0 else 0)
        remainder -= 1

        # Reduziere die Anzahl der zu ziehenden Zeilen auf die verfügbare Anzahl, falls nötig
        available_count = len(author_df[author_df['Titel'] == title])
        rows_to_select = min(count, available_count)

        # Wähle die Zeilen für diesen Titel
        selected_rows.append(author_df[author_df['Titel'] == title].sample(rows_to_select, random_state=1))

    # Kombiniere die ausgewählten Zeilen
    selected_rows_df = pd.concat(selected_rows)

    # Sicherstellen, dass jeder Autor n Zeilen hat
    selected_counts = selected_rows_df.groupby('Author').size()  # Zähle die Zeilen pro Autor

    # Füge Zeilen hinzu, falls ein Autor weniger als n Zeilen hat
    for author in selected_counts.index:
        if selected_counts[author] < n:
            needed_rows = n - selected_counts[author]
            remaining_rows = df[df['Author'] == author].drop(selected_rows_df[selected_rows_df['Author'] == author].index)
            extra_rows = remaining_rows.sample(needed_rows, random_state=1)
            selected_rows_df = pd.concat([selected_rows_df, extra_rows])

            selected_counts = selected_rows_df.groupby('Author').size()  # Zähle die Zeilen pro Autor

    # Kombiniere die ausgewählten Zeilen und gib sie zurück
    return selected_rows_df.reset_index(drop=True)

# Erstelle ein DataFrame, in dem genau n Zeilen pro Autor enthalten sind
filtered_df = pd.concat([select_balanced_rows(filtered_df, author) for author in filtered_df['Author'].unique()])

# Speichere das gefilterte und balancierte DataFrame
filtered_df.to_csv('LiteraturDatenMitTexten_prepared_balanced.csv', index=False)




# Funktion, um genau 2 zufällige Zeilen pro Autor zu wählen, möglichst mit unterschiedlichen Titeln --> Für Testdaten
def select_test_data(df, author):
    # Filtere die Zeilen des aktuellen Autors
    author_df = df[df['Author'] == author]

    # Hole die Titel für den aktuellen Autor
    unique_titles = author_df['Titel'].unique()

    # Falls der Autor nur einen Titel hat, wähle zwei zufällige Zeilen davon (wird nicht verhindert)
    if len(unique_titles) < 2:
        test_rows = author_df.sample(2, random_state=1)
    else:
        # Wähle zufällig zwei verschiedene Titel
        selected_titles = pd.Series(unique_titles).sample(2, random_state=1).values
        test_rows = pd.concat([author_df[author_df['Titel'] == title].sample(1, random_state=1) for title in selected_titles])

    return test_rows

print("Länge filtered_df: ", len(filtered_df))



# Wähle genau 2 Zeilen pro Autor für den Test-Datensatz aus
test_data = pd.concat([select_test_data(filtered_df, author) for author in filtered_df['Author'].unique()])
print("Länge Testdaten: ", len(test_data))

# Entferne die ausgewählten Test-Daten aus dem ursprünglichen DataFrame, um die Trainingsdaten zu erstellen
train_data = filtered_df[~filtered_df['Textausschnitt'].isin(test_data['Textausschnitt'])]
print("Länge Trainingsdaten: ", len(train_data))

# Sicherstellen, dass die Länge von train_data korrekt ist
expected_train_length = len(filtered_df) - len(test_data)
assert len(train_data) == expected_train_length, f"Fehler: Erwartete Länge von train_data ist {expected_train_length}, aber tatsächlich {len(train_data)}"

# Mische die Reihenfolge in den Trainings- und Testdaten
train_data = train_data.sample(frac=1, random_state=1).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=1).reset_index(drop=True)

# Speichere die Test-Daten in eine CSV-Datei
test_data.to_csv("test_mitLabel.csv", index=False)

# Speichere nur die 'Textausschnitt'-Spalte in test_ohneLabel.csv
test_data['Textausschnitt'].to_csv("test_ohneLabel.csv", index=False, header=False)

# Speichere die Trainingsdaten in eine CSV-Datei
train_data.to_csv("train.csv", index=False)

# Speichere die Trainingsdaten mit nur der Autor-ID und dem Textausschnitt in eine CSV-Datei
train_data[['Author_id', 'Textausschnitt']].to_csv("train_ohneMetadata.csv", index=False, header=False)
