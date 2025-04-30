from pathlib import Path
import pandas as pd
from datetime import timedelta
from dateutil import parser
from pandas import Timestamp

BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
SLEEP_DATA_FILE = DATA_DIR / "Sommeil_Test.csv"

def parse_time_ts(ts):
    """
    Convertit une chaîne horaire en objet datetime.time grâce à dateutil.

    Cette fonction prend en entrée :
      - ts : une chaîne représentant une heure (ex. "03:56 AM") ou une valeur manquante.
    Elle retourne :
      - un objet datetime.time si le parsing réussit,
      - None si la chaîne est invalide ou manquante.

    Paramètres
    ----------
    ts : str or float or NaT
        Chaîne au format 12 heures (avec AM/PM) ou valeur manquante (NaN/NaT).

    Retour
    ------
    datetime.time or None
        L'heure extraite, ou None en cas d’erreur ou de valeur manquante.
    """
    # Parsing des heures via dateutil.parser
    if pd.isna(ts):
        return None
    try:
        return parser.parse(ts).time()
    except Exception:
        return None

    
def combine(row, time_col):
    """
    Construit un pd.Timestamp en combinant la date et l’heure d’une ligne.

    Cette fonction prend une ligne de DataFrame `row` et le nom d’une colonne
    contenant un objet `datetime.time`, puis crée un `pd.Timestamp` avec la
    date issue de `row['date']` et l’heure issue de `row[time_col]`. Si
    l’heure est absente (`None`), retourne `pd.NaT`.

    Paramètres
    ----------
    row : pandas.Series
        Une ligne du DataFrame contenant au moins :
          - 'date' : un objet datetime.date ou pd.Timestamp
          - `time_col` : un objet datetime.time ou None
    time_col : str
        Nom de la colonne dans `row` qui contient l’objet `time`.

    Retour
    ------
    pandas.Timestamp or pd.NaT
        Un timestamp complet (date + heure), ou `NaT` si `row[time_col]` est None.
    """
    t = row[time_col]
    if t is None:
        return pd.NaT
    return Timestamp(
        year=row['date'].year,
        month=row['date'].month,
        day=row['date'].day,
        hour=t.hour,
        minute=t.minute
    )

def calculer_statistiques_sommeil (df) : 
    """
    Calcule les statistiques de sommeil à partir d'un DataFrame.

    Cette fonction s'attend à ce que le DataFrame contienne une colonne 'duree'
    exprimée en minutes. Elle filtre les nuits valides (duree non nulle) et
    renvoie un tuple contenant :
      - totales     : nombre total de nuits analysées (int)
      - normales    : nombre de nuits dont la durée est entre 420 et 540 min (int)
      - courtes     : nombre de nuits < 420 min (int)
      - longues     : nombre de nuits > 540 min (int)
      - moyenne     : durée moyenne en minutes (float)
      - mediane     : durée médiane en minutes (float)
      - minimale    : durée minimale en minutes (float)
      - maximale    : durée maximale en minutes (float)
      - ecart_type  : écart-type des durées en minutes (float)

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame contenant au moins la colonne 'duree' (float).

    Retour
    ------
    tuple
        (totales, normales, courtes, longues,
         moyenne, mediane, minimale, maximale, ecart_type)
    """
    df_clean = df[df['duree'].notna()]
    totales = len(df_clean)
    normales = df_clean[(df_clean['duree'] >= 420) & (df_clean['duree'] <= 540)].shape[0]
    courtes  = df_clean[df_clean['duree'] < 420].shape[0]
    longues  = df_clean[df_clean['duree'] > 540].shape[0]
    avg    = df_clean['duree'].mean()
    med    = df_clean['duree'].median()
    mina   = df_clean['duree'].min()
    maxa   = df_clean['duree'].max()
    stddev = df_clean['duree'].std()
    return totales,normales,courtes,longues,avg,med,mina,maxa,stddev

def afficher_rapport(stats_sommeil ,date_min,date_max):
    """
    Affiche un rapport de sommeil à l’écran, avec statistiques et période.

    Cette fonction reçoit :
      - stats_sommeil : tuple de 9 valeurs retourné par `calculer_statistiques_sommeil`,
          dans l’ordre (totales, normales, courtes, longues,
          moyenne, mediane, minimale, maximale, ecart_type).
      - date_min      : date de début de la période (str ou pd.Timestamp).
      - date_max      : date de fin de la période (str ou pd.Timestamp).

    Elle imprime :
      - Le nombre total de nuits analysées et leur répartition (courtes/normales/longues).
      - Les durées moyenne, médiane, minimale, maximale et l’écart-type,
        le tout en minutes et en heures approximatives.
      - Une légende expliquant médiane et écart-type.

    Paramètres
    ----------
    stats_sommeil : tuple
        Résultat de `calculer_statistiques_sommeil(df)` :
        (totales, normales, courtes, longues,
         moyenne, mediane, minimale, maximale, ecart_type).
    date_min : str or pandas.Timestamp
        Date de début de la période à afficher.
    date_max : str or pandas.Timestamp
        Date de fin de la période à afficher.

    Returns
    -------
    None
        Le rapport est directement imprimé sur la sortie standard.
    """
    print(f"\n--- Rapport Sommeil du {date_min} au {date_max} ---\n")
    print(f"Nombre de nuits analysées : {stats_sommeil [0]}")
    print(f"  - normales : {stats_sommeil [1]}")
    print(f"  - courtes  : {stats_sommeil  [2]}")
    print(f"  - longues  : {stats_sommeil [3]}\n")
    print(f"Durée moyenne   : {stats_sommeil [4]:.0f} min (~{stats_sommeil [4]/60:.1f} h)")
    print(f"Durée médiane   : {stats_sommeil [5]:.0f} min (~{stats_sommeil [5]/60:.1f} h)")
    print(f"Durée minimale  : {stats_sommeil [6]:.0f} min (~{stats_sommeil [6]/60:.1f} h)")
    print(f"Durée maximale  : {stats_sommeil [7]:.0f} min (~{stats_sommeil [7]/60:.1f} h)")
    print(f"Ecart-type      : {stats_sommeil [8]:.0f} min")
    print("\n(1) Médiane moins sensible aux valeurs extrêmes")
    print("(2) Ecart-type faible → sommeil régulier\n")
    
import pandas as pd
import re
from pathlib import Path
from datetime import timedelta
from dateutil import parser
from pandas import Timestamp

def prepare_sleep_df(path):
    """
    Charge et prépare un DataFrame de sommeil depuis un fichier CSV.

    Étapes réalisées :
    1. Lecture du CSV depuis `path`.
    2. Nettoyage des noms de colonnes (remplacement des NBSP).
    3. Sélection et renommage des colonnes d'intérêt.
    4. Conversion de la colonne 'date' en datetime.
    5. Copie des heures brutes et normalisation des indicateurs manquants '--'.
    6. Nettoyage des chaînes d'heure :
       - suppression des espaces multiples et invisibles,
       - retrait des apostrophes,
       - padding des heures à un chiffre.
    7. Extraction du pattern horaire (HH:MM AM/PM) par regex.
    8. Parsing des heures en objets `time` via `dateutil.parser`.
    9. Combinaison date + time → `Timestamp`, ajustement si lever ≤ coucher.
    10. Calcul de la durée en minutes, et extraction de composantes heures/minutes.

    Paramètre
    ---------
    path : str or pathlib.Path
        Chemin vers le fichier CSV de sommeil.

    Retour
    ------
    pandas.DataFrame
        Le DataFrame enrichi des colonnes :
        ['date', 'coucher', 'lever', 'duree', 'heures', 'minutes', …]
    """
    # 1) Lecture
    df = pd.read_csv(path)

    # 2) Nettoyage des noms de colonnes
    df.columns = df.columns.str.replace('\u00A0', ' ', regex=False)

    # 3) Sélection & renommage
    df = (
        df
        .drop(columns=['Durée'], errors='ignore')
        .rename(columns={
            'Sommeil 4 semaines': 'date',
            'Heure de coucher'  : 'coucher',
            'Heure de lever'    : 'lever'
        })
    )

    # 4) Conversion date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 5) Copie des heures brutes & normalisation des "--"
    df['raw_coucher'] = df['coucher'].astype(str).replace('--', pd.NA)
    df['raw_lever']   = df['lever'  ].astype(str).replace('--', pd.NA)

    # 6) Nettoyage et padding
    for col in ['raw_coucher', 'raw_lever']:
        df[col] = (
            df[col]
            .str.replace(r'\s+', ' ', regex=True)   # espaces multiples → espace
            .str.strip()                            # trim
            .str.strip("'")                         # retirer apostrophes
            .str.replace(r'^(\d):', r'0\1:', regex=True)  # padding x: → 0x:
        )

    # 7) Extraction du pattern horaire
    time_pattern = r'(\d{2}:\d{2}\s?[AP]M)'
    df['only_coucher'] = df['raw_coucher'].str.extract(time_pattern, expand=False).str.strip()
    df['only_lever']   = df['raw_lever'  ].str.extract(time_pattern, expand=False).str.strip()

    # 8) Parsing en `time`
    def _parse(ts):
        if pd.isna(ts):
            return None
        try:
            return parser.parse(ts).time()
        except:
            return None

    df['time_coucher'] = df['only_coucher'].apply(_parse)
    df['time_lever']   = df['only_lever'].apply(_parse)

    # 9) Combinaison date + time → Timestamp
    def _combine(r, col):
        t = r[col]
        if t is None:
            return pd.NaT
        return Timestamp(
            year = r['date'].year,
            month= r['date'].month,
            day  = r['date'].day,
            hour = t.hour,
            minute=t.minute
        )

    df['time_coucher'] = df['only_coucher'].apply(parse_time_ts)
    df['time_lever']   = df['only_lever'].apply(parse_time_ts)

    # Ajustement nuits après minuit
    mask = df['lever'] <= df['coucher']
    df.loc[mask, 'lever'] += timedelta(days=1)

    # 10) Calcul durée et composantes
    df['duree'] = (df['lever'] - df['coucher']).dt.total_seconds() / 60
    comps = (df['lever'] - df['coucher']).dt.components
    df['heures']  = comps.days * 24 + comps.hours
    df['minutes'] = comps.minutes

    return df


df = prepare_sleep_df(SLEEP_DATA_FILE)
stats_sommeil  = calculer_statistiques_sommeil (df)

afficher_rapport(stats_sommeil ,df['date'].min().strftime('%A %d %B %Y'),df['date'].max().strftime('%A %d %B %Y'))
#affichage du dataframe
df_display = df[~((df['coucher'].isna()) | (df['lever'].isna()))].copy() # Filtre sur une copie
# Crée une colonne formatée pour l'affichage SEULEMENT dans cette copie
df_display['duree_h'] = "~" + (df_display['duree'] / 60).round(1).astype(str) + " h"
print(df_display[["date",'only_coucher','only_lever','duree_h','heures','minutes']])
