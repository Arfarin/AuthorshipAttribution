import pandas as pd
import matplotlib.pyplot as plt
import os


# CSV-Datei einlesen
datei_pfad = 'LiteraturDatenMitTexten_prepared.csv'
df = pd.read_csv(datei_pfad, delimiter=';')

# Entfernen von Leerzeichen am Anfang und Ende der Autorennamen
df['Author'] = df['Author'].str.strip()


# Alle Datensätze mit Genre "Lehrbuch" entfernen
df = df[df['Genre'] != 'Lehrbuch']

# Anzahl der Textausschnitte, bei denen "Textausschnitt online kostenfrei verfügbar" "ja" ist
anzahl_kostenfrei = df['Textausschnitt online kostenfrei verfügbar'].str.lower().value_counts().get('ja', 0)
# Ergebnis anzeigen
print(f"Anzahl der Textausschnitte, die online kostenfrei verfügbar sind: {anzahl_kostenfrei}")

# Anzahl der verschiedenen Autoren
anzahl_autoren = df['Author'].nunique()
print(f"Anzahl der verschiedenen Autoren: {anzahl_autoren}")

# Anzahl der Textausschnitte pro Autor
excerpts_pro_autor = df.groupby('Author')['Textausschnitt'].nunique().sort_values(ascending=False)
print("Anzahl der Textausschnitte pro Autor:")
print(excerpts_pro_autor)
# Durchschnittliche Anzahl an Textausschnitten pro Autor berechnen
durchschnitt_excerpts_pro_autor = excerpts_pro_autor.mean()
print("Durchschnittl. Anzahl Textausschnitte pro Autor: ", durchschnitt_excerpts_pro_autor)

# Anzahl der verschiedenen Titel pro Autor
titel_pro_autor = df.groupby('Author')['Titel'].nunique().sort_values(ascending=False)
print("Anzahl der verschiedenen Titel pro Autor:")
print(titel_pro_autor)
durchschnitt_anz_titel_pro_autor = titel_pro_autor.mean()
print("Durchschnittl. Anzahl Titel pro Autor: ", durchschnitt_anz_titel_pro_autor)


# Wortanzahl in der Spalte "Textausschnitt" berechnen
df['Wortanzahl'] = df['Textausschnitt'].str.split().str.len()

# Min, Max und Average Wortanzahl der Textausschnitte berechnen
min_wortanzahl = df['Wortanzahl'].min()
max_wortanzahl = df['Wortanzahl'].max()
avg_wortanzahl = df['Wortanzahl'].mean()

print(f"\nMinimale Wortanzahl: {min_wortanzahl}")
print(f"Maximale Wortanzahl: {max_wortanzahl}")
print(f"Durchschnittliche Wortanzahl: {avg_wortanzahl}")

# Durchschnittliche Wortanzahl pro Autor berechnen
durchschnittliche_wortanzahl_pro_autor = df.groupby('Author')['Wortanzahl'].mean().round(1).sort_values(ascending=False)
print("Durchschnittliche Wortanzahl pro Autor:")
print(durchschnittliche_wortanzahl_pro_autor)

# Duplikate basierend auf "Titel" entfernen
eindeutige_titel = df.drop_duplicates(subset='Titel')

# Genre-Übersetzungen definieren
genre_mapping = {
    'Kinderbuch/Jugendliteratur': 'Childrens book',
    'Roman': 'Novel',
    'Novelle': 'Novel',
    'Brief': 'Letter',
    'Kunstmärchen': 'Tale',
    'Erzählung': 'Tale'
}

# Genres übersetzen
eindeutige_titel['Genre_english'] = eindeutige_titel['Genre'].replace(genre_mapping)
df['Genre_english'] = df['Genre'].replace(genre_mapping)

# Häufigkeit der Genres berechnen
genre_counts = eindeutige_titel['Genre_english'].value_counts()

# Gesamtanzahl der einzigartigen Titel
gesamt_eindeutige_titel = genre_counts.sum()

# Anteil der Genres berechnen
genre_anteile = (genre_counts / gesamt_eindeutige_titel) * 100
print("Genres und ihre Anteile (basierend auf einzigartigen Titeln):")
for genre, anteil in genre_anteile.items():
    print(f"{genre}: {anteil:.2f}%")
    
def set_mpl_configurations():
    
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title    

set_mpl_configurations()
# Barchart erstellen
plt.figure(figsize=(10, 6))
genre_counts.plot(kind='bar')
#plt.title('Frequency of genres (based on unique titles)')
plt.xlabel('Genre')
plt.ylabel('Number of titles')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # Für bessere Darstellung
plt.box(False)

# Prozentualen Anteil berechnen
gesamt_titel = gesamt_eindeutige_titel.sum()
anteile = (genre_counts / gesamt_titel) * 100

# Prozentuale Anteile in die Balken schreiben
for i in range(len(genre_counts)):
    plt.text(i, genre_counts[i] + 0.5, f'{anteile.iloc[i]:.1f}%', ha='center')


# Sicherstellen, dass das Verzeichnis 'images' existiert
os.makedirs('images', exist_ok=True)

# Diagramm als Bild speichern
bild_pfad = 'images/genre_haufigkeit.png'
plt.savefig(bild_pfad)

def gib_tabelle_zu_autoren():
    # Anzahl der Titel pro Autor
    anzahl_titel = eindeutige_titel.groupby('Author')['Titel'].nunique()

    # Anzahl der Textausschnitte pro Autor
    anzahl_textausschnitte = df.groupby('Author')['Textausschnitt'].nunique()
    
    # Durchschnittliche Wortanzahl in Textausschnitten pro Autor berechnen
    durchschnittliche_wortanzahl = df.groupby('Author')['Wortanzahl'].mean()

    tabelle = pd.DataFrame({
        'Anzahl Titel': anzahl_titel,
        'Anzahl Textausschnitte': anzahl_textausschnitte,
        'Durchschnittliche Wortanzahl in Textausschnitten': durchschnittliche_wortanzahl_pro_autor
    })

    # Indizes zurücksetzen, um die Autoren als Spaltenüberschrift zu haben
    tabelle = tabelle.reset_index()

    # Tabelle anzeigen
    print(tabelle)
    
def gib_bild_textausschnitte_genre():
    # Anzahl der Textausschnitte pro Genre berechnen
    textausschnitte_pro_genre = df.groupby('Genre_english')['Textausschnitt'].nunique().sort_values(ascending=False)

    # Barchart erstellen
    plt.figure(figsize=(10, 6))
    textausschnitte_pro_genre.plot(kind='bar')
    #plt.title('Number of excerpts per genre')
    plt.xlabel('Genre')
    plt.ylabel('Number of text excerpts')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Für bessere Darstellung
    plt.box(False)
    
    # Prozentualen Anteil berechnen
    gesamt_textausschnitte = textausschnitte_pro_genre.sum()
    anteile = (textausschnitte_pro_genre / gesamt_textausschnitte) * 100

    # Prozentuale Anteile in die Balken schreiben
    for i in range(len(textausschnitte_pro_genre)):
        plt.text(i, textausschnitte_pro_genre.iloc[i] + 0.5, f'{anteile.iloc[i]:.1f}%', ha='center')

    # Sicherstellen, dass das Verzeichnis 'images' existiert
    os.makedirs('images', exist_ok=True)

    # Diagramm als Bild speichern
    bild_pfad = 'images/textausschnitte_pro_genre.png'
    plt.savefig(bild_pfad)
    
def authors_with_en_and_de():
    # Autoren mit Textausschnitten in "de" und "en" finden
    autoren_de = set(df[df['Sprache'] == 'de']['Author'])
    autoren_en = set(df[df['Sprache'] == 'en']['Author'])

    # Schnittmenge der Autoren
    gemeinsame_autoren = autoren_de.intersection(autoren_en)

    # Anzahl der gemeinsamen Autoren ausgeben
    anzahl_gemeinsame_autoren = len(gemeinsame_autoren)
    print(f"Anzahl der Autoren mit Textausschnitten in beiden Sprachen (de und en): {anzahl_gemeinsame_autoren}")    

gib_tabelle_zu_autoren()
gib_bild_textausschnitte_genre()
#authors_with_en_and_de()


