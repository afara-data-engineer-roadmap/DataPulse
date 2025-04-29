from pathlib import Path 
import pandas as pd 

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"
SLEEP_DATA_FILE = DATA_DIR / "Sommeil.csv"

def afficher_rapport(df) :
    print("\n--- Rapport Sommeil (Basique) ---\n")
    avg_duration = df['duree'].mean()
    max_duration = df['duree'].max()
    min_duration = df['duree'].min()
    if not pd.isna(avg_duration):
        print(f"Durée moyenne du sommeil   : {avg_duration:.0f} minutes (~{avg_duration/60:.1f} heures)")
        print(f"Durée minimale             : {min_duration:.0f} minutes (~{min_duration/60:.1f} heures)")
        print(f"Durée maximale             : {max_duration:.0f} minutes (~{max_duration/60:.1f} heures)")
        print("\n---------------------------------\n")
#Lecture du fichier
df = pd.read_csv(SLEEP_DATA_FILE)
#Suppression des espaces insécables dans les noms de colonne
df.columns = df.columns.str.replace('\u00A0', ' ', regex=False)

#Suppression des colonnes inutiles et renommage
df = df.drop(columns=['Durée'])
df = df.rename(columns={
                        'Sommeil 4 semaines': "date" ,
                        'Heure de coucher': "coucher" ,
                        'Heure de lever': "lever" 
                        }
                )
#Suppression des lignes non pertinentes
df = df[~((df['Heure de coucher'] == '--') | (df['Heure de lever'] == '--'))]

#Changement des types de colonnes contentant des dates ou des horaires
df["date"] = pd.to_datetime(df["date"])
df["coucher"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " +df["coucher"],errors="coerce")
df["lever"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " +df["lever"],errors="coerce")

#Mise à jour de la date en fonction de l'heure 
mask = df["lever"] <= df["coucher"]
df.loc[mask, "lever"] += pd.Timedelta(days=1)


df["duree"] = df["lever"] - df["coucher"]

#Calcul du nombre de minutes et du nombre d'heures de sommeil
comps = df["duree"].dt.components
df["heures"] = comps.days * 24 + comps.hours
df["minutes"] = comps.minutes
#calul de la durée de sommeil en minutes
df["duree"] = df["duree"].dt.total_seconds() / 60
#changement de format des colonnes coucher et lever datetime => str et 12h => 24h
df["coucher"] = df["coucher"].dt.strftime("%H:%M")
df["lever"] = df["lever"].dt.strftime("%H:%M")

afficher_rapport(df)

print(df)


