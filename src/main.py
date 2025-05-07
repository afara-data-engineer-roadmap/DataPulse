from pathlib import Path
import pandas as pd
from datetime import timedelta
from dateutil import parser
from pandas import Timestamp
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt 

import seaborn as sns


BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
SLEEP_DATA_FILE = DATA_DIR / "Sommeil.csv"
COLS_REQ_STATS = ['date','coucher','lever','qualite','raw_coucher','raw_lever','only_coucher','only_lever','time_lever','duree','heures','minutes','weekend']

def visualiser_coucher_vs_duree(df, *, avec_regression: bool = True, ax=None):
    
    # S'assurer que les colonnes nécessaires existent et sont valides
    data_plot = df[['mins_coucher', 'duree']].dropna()

    if data_plot.empty:
        logger.info("Données insuffisantes pour visualiser heure de coucher vs durée.")
        return
    
    # 2. Nuage de points
    sns.scatterplot(data=data_plot, x='mins_coucher', y='duree',
                   alpha=0.6, edgecolor="w", ax=ax)
    
    # 3. Régression linéaire (optionnelle)
    if avec_regression:
        sns.regplot(data=data_plot, x='mins_coucher', y='duree',
                    scatter=False, ax=ax, line_kws={'linewidth': 2})
        
    # 4. Mise en forme
    ax.set(title="Durée du sommeil en fonction de l'heure de coucher",
           xlabel="Heure de coucher (minutes depuis minuit)",
           ylabel="Durée du sommeil (minutes)")
    ax.grid(True)

    
    # 5. Affichage
    plt.tight_layout()
    
    logger.info("nuage de points coucher vs durée (+ régression) OK.")
    try :     
        plt.show()
        logger.info("Scatter coucher vs durée (+ régression) affiché avec succès.")
    except Exception as e :
        logger.info("Echec lors de la visualisation Durée du Sommeil en fonction de l\'Heure de Coucher : %s", e)
    
    
def visualiser_evolution_duree(df,*,window : int = 7 ,ax = None):
    
    """affiche la durée de sommeil et la moyenne mobile centrée sur 7 jours par défaut 
    Parameters 
    
    df : pd.DataFrame - contient 'date' et 'duree'
    window : int - taille de la fenêtre glissante (en jours)
    ax : matplotlib.axes.Axes ou None
        Axe matplotlib existant sur lequel tracer (utile pour superposer plusieurs courbes 
        sur un même graphique). Si None, une nouvelle figure est créée automatiquement.
    
    """
    duree_valide = df[['date', 'duree']].dropna() # Garde date et duree, enlève les NaN de duree
    df_ord = duree_valide.sort_values('date')
    
    if ax is None:
        fig,ax = plt.subplots(figsize=(20,16))


    if duree_valide.empty:
        logger.info("Aucune donnée de durée valide pour visualiser l'évolution.")
        return

    sns.lineplot(ax = ax,x ='date', y='duree', data=df_ord, marker='o',label = "Durée nuit", alpha = 0.4) # 'o' ajoute des points sur la ligne
    rolling = (
        df_ord.set_index('date')['duree']
              .rolling(window=window, center=True, min_periods=window//2)
              .mean()
    )
    ax.plot(rolling.index, rolling.values,
            linewidth=2, label=f"Moyenne mobile ({window}j)")
    ax.set_title("Évolution de la durée de sommeil")
    ax.set_xlabel("Date")
    ax.set_ylabel("Durée du sommeil (minutes)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    try : 
        plt.show()
    except Exception as e :
        logger.info("Echec lors de la visualisation  Évolution de la Durée du Sommeil au Fil du Temps:  %s", e)
    logger.info("Visualisation Évolution de la durée du sommeil créée avec succès.")
    
def visualiser_distribution_duree(df) : 
    
    duree_valide = df['duree'].dropna()
    
    if duree_valide.empty : 
        logger.info("aucune donnnée de durée valide pour générer la visualisation.")
        return 
    
    plt.figure (figsize = (20,16))
    sns.histplot(duree_valide,kde =True ,bins=10)
    #plt.hist(duree_valide, bins=10, edgecolor='black', alpha=0.7)
    
    plt.title('Distribution de la durée du sommeil')
    plt.xlabel('Durée du sommeil (mns)')
    plt.ylabel('Fréquence (nbr nuits)')
    plt.grid(True)
    try : 
        plt.show()
    except Exception as e :
        logger.info("Echec lors de la visualisation Distribution de la durée du sommeil : %s", e)

    logger.info("Visualisation Distribution de la durée du sommeil créée avec succès.")



def configure_logging(log_path="mon_script.log"):
    """
    Configure un logger global :
     - Vide d'abord tous les handlers du root logger
     - Ajoute un RotatingFileHandler (5 Mo max, 5 backups) en UTF-8-BOM, niveau DEBUG
     - Ajoute un StreamHandler console niveau INFO
    """
    root = logging.getLogger()
    # 1) on enlève tous les anciens handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    # 2) handler fichier rotatif
    try:
        fh = RotatingFileHandler(
            log_path,
            maxBytes=5_000_000,
            backupCount=5,
            mode="w",
            encoding="utf-8-sig"      # UTF-8 avec BOM
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
    except PermissionError as e:
        logger.error("Pas de droit d'écriture sur le log : %s", e)     
    # 3) handler console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # 4) on fixe le niveau root et on attache
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

configure_logging(DATA_DIR/"mon_script.log")
logger = logging.getLogger(__name__)

def controle_validite(row) : 
    invalidite = "valide"
    if row["Durée"] == '--' or pd.isna(row["Durée"]) :
        invalidite =  "-Durée manquante"
    if row["Heure de coucher"] == '--' or pd.isna(row["Heure de coucher"]) : 
        invalidite +=   "-Heure de coucher manquante"
    if row["Heure de lever"] == '--' or pd.isna(row["Heure de lever"]) : 
        invalidite += "-Heure de lever manquante"
    return invalidite


def lire_fichier_pandas(chemin_fichier, encodages=('utf-8', 'latin-1', 'cp1252'), separateurs=(';', ',', '\t', '|', ':')):
    """
    Lit un fichier tabulaire (CSV/TXT) en testant plusieurs encodages et séparateurs avec pandas.

    Paramètres :
    - chemin_fichier (str) : chemin du fichier à lire
    - encodages (tuple) : liste d'encodages à tester
    - separateurs (tuple) : liste de séparateurs possibles

    Retourne :
    - dict avec 'dataframe' (pd.DataFrame), 'encodage_utilisé' et 'séparateur_utilisé'
    - None si échec
    """
    for encodage in encodages:
        for sep in separateurs:
            try:
                df = pd.read_csv(chemin_fichier, encoding=encodage, sep=sep)
                
                # Si le fichier n'est pas vide
                if not df.empty and len(df.columns) >1:
                    logger.info("fichier %s lu avec succès " ,  chemin_fichier )
                    return {
                        'dataframe': df,
                        'encodage_utilisé': encodage,
                        'séparateur_utilisé': sep
                    }
                    
            except UnicodeDecodeError as e:
                logger.warning("Erreur d'encodage avec %s : %s " ,  encodage, e )
            except pd.errors.ParserError as e:
                logger.warning("Erreur de séparateur avec %s : %s " ,  sep, e )
                continue
            except Exception as e:
                logger.warning("Erreur inatendue avec %s : %s " ,  sep, e )
                continue
    logger.critical("Échec : Impossible de lire %s avec les encodages et séparateurs fournis." ,  chemin_fichier)        
    return None

def mins_to_hhmm(total_minutes):
    """
   Convertit un nombre de minutes depuis minuit en chaîne 'HH:MM'.

   Parameters
   ----------
   total_minutes : float or int or NaN
       Nombre de minutes écoulées depuis 00:00. Peut être fractionnaire ou NaN.

   Returns
   -------
   str
       Chaîne formatée 'HH:MM' ou 'N/A' si la valeur est manquante.
   """
   
    if pd.isna(total_minutes):
        return "N/A"
    total_minutes = int(total_minutes)
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


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
    except Exception as e:
        logger.debug("Impossible de parser %r en time : %s", ts, e)
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
    # NOTE: for large DataFrames, consider vectorizing with pd.to_datetime(...)

    t = row[time_col]
    if t is None:
        logger.debug("Aucune heure détectée dans '%s' pour la date %s",time_col, row.get('date'))
        return pd.NaT
    return Timestamp(
        year=row['date'].year,
        month=row['date'].month,
        day=row['date'].day,
        hour=t.hour,
        minute=t.minute
    )

def calculer_stats_globales (df) : 
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
    COLS_REQ_STATS = ['date','coucher','lever','qualite','raw_coucher','raw_lever','only_coucher','only_lever','time_lever','duree','heures','minutes','weekend']

    logger.info("Début du calcul des stats globales sur %d lignes", len(df))

   
    df_clean = df[df['duree'].notna()] # La ligne clé

   
    try :
        if df_clean.empty:
            logger.warning("Aucune donnée valide pour le calcul des statistiques")
            return (0,0,0,0,0,0,0,0,0)
        totales = len(df_clean)
        normales = df_clean[(df_clean['duree'] >= 420) & (df_clean['duree'] <= 540)].shape[0]
        courtes  = df_clean[df_clean['duree'] < 420].shape[0]
        longues  = df_clean[df_clean['duree'] > 540].shape[0]
        avg    = df_clean['duree'].mean()
        med    = df_clean['duree'].median()
        mina   = df_clean['duree'].min()
        maxa   = df_clean['duree'].max()
        stddev = df_clean['duree'].std()
        logger.info("Stats globales — totales: %d, moy: %.1f, médiane: %.1f, min: %.1f, max: %.1f",totales, avg, med, mina, maxa)
        return totales,normales,courtes,longues,avg,med,mina,maxa,stddev
    except Exception as e : 
        logger.exception("Erreur lors du calcul des stats globales : %s" , e) 
        return (0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

def calculer_stats_par_periode(df):
    """
    Retourne (moyenne_semaine, moyenne_weekend).
    """
    logger.info("Début du calcul des stats par période")

    try :
        moyennes = df.groupby(df['date'].dt.weekday >= 5)['duree'].mean()
        moyennes = moyennes.reindex([False, True], fill_value=0)
        logger.info("Stats periodiques — weekend:  %.1f, semaine: %.1f", moyennes[False], moyennes[True])
        
        return moyennes[False], moyennes[True]
    except Exception as e : logger.exception ("Erreur lors du calcul des stats par pèriode : %s ", e)

def calculer_heures_medianes(df):
    """
    Retourne (median_coucher, median_lever).
    """
    logger.info("Début du calcul des heures médianes")

    try : 
        median_coucher = df['mins_coucher'].dropna().median()
        median_lever = df['mins_lever'].dropna().median()
        logger.info("Heures médianes calculées — coucher: %d min, lever: %d min", median_coucher, median_lever)
        return median_coucher, median_lever
    
    except Exception as e : 
        logger.exception("Erreur lors du calcule des heures médianes", e )
        return 0.0,0.0
        


def afficher_rapport(stats_sommeil ,date_min,date_max):
    """
    Affiche un rapport de sommeil à l’écran, avec statistiques et période.

    Cette fonction reçoit :
      - stats_sommeil : tuple de 9 valeurs 
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
    print(f"Nombre de nuits analysées   : {stats_sommeil [0]}")
    print(f"  - normales                : {stats_sommeil [1]}")
    print(f"  - courtes                 : {stats_sommeil  [2]}")
    print(f"  - longues                 : {stats_sommeil [3]}\n")
    print(f"Durée moyenne               : {stats_sommeil [4]:.0f} min (~{stats_sommeil [4]/60:.1f} h)")
    print(f"Durée médiane(1)            : {stats_sommeil [5]:.0f} min (~{stats_sommeil [5]/60:.1f} h)")
    print(f"Durée minimale              : {stats_sommeil [6]:.0f} min (~{stats_sommeil [6]/60:.1f} h)")
    print(f"Durée maximale              : {stats_sommeil [7]:.0f} min (~{stats_sommeil [7]/60:.1f} h)")
    print(f"Ecart-type                  : {stats_sommeil [8]:.0f} min\n")
    print(f"Durée moyenne en semaine    : {stats_sommeil [9]:.0f} min (~{stats_sommeil[9]/60:.1f} h)")
    print(f"Durée moyenne le week-end   : {stats_sommeil [10]:.0f} min (~{stats_sommeil[10]/60:.1f} h)\n")
    print(f"Heure médiane de coucher(1) : {mins_to_hhmm(stats_sommeil [11])}")
    print(f"Heure médiane de lever(1)   : {mins_to_hhmm(stats_sommeil [12])}")
    print("\n---------------------------------------------------\n")
    print("\n(1) Médiane moins sensible aux valeurs extrêmes")
    print("(2) Ecart-type faible → sommeil régulier\n")

    


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
    try:

        logger.info("Début de prepare_sleep_df pour %s", path)
        
        #  Lecture
        df =lire_fichier_pandas(path)["dataframe"]
        if df is None:
            logger.critical("Abandon de prepare_sleep_df : échec de lecture de %s", path)
            return None
        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.replace('\u00A0', ' ', regex=False)
        
        #controle qualite
        df["qualite"] = df.apply(controle_validite, axis=1)
        logger.debug("Avant drop/rename : shape=%s, colonnes=%s", df.shape, df.columns.tolist())

        # Sélection & renommage
        df = (
            df
            .drop(columns=['Durée'], errors='ignore')
            .rename(columns={
                'Sommeil 4 semaines': 'date',
                'Heure de coucher'  : 'coucher',
                'Heure de lever'    : 'lever'
            })
        )
        logger.debug("Après drop/rename  : shape=%s, colonnes=%s", df.shape, df.columns.tolist())

        # 4) Conversion date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().any():
            logger.warning("Certaines dates n'ont pas pu être parsées")
        if df['date'].isna().all():
            logger.warning("Toutes les dates sont NaT après conversion pour %s", path)
        
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
        
        # conversion time
        df['time_coucher'] = df['only_coucher'].apply(parse_time_ts)
        df['time_lever']   = df['only_lever'].apply(parse_time_ts)
        
        # reconstruction 
        df['coucher'] = df.apply(combine, args=('time_coucher',), axis=1)
        df['lever']   = df.apply(combine, args=('time_lever',), axis=1)
    
        # Ajustement nuits après minuit
        mask = df['lever'] <= df['coucher']
        df.loc[mask, 'lever'] += timedelta(days=1)
    
        # Calcul durée et composantes durée
        df['duree'] = (df['lever'] - df['coucher']).dt.total_seconds() / 60
        comps = (df['lever'] - df['coucher']).dt.components
        df['heures']  = comps.days * 24 + comps.hours
        df['minutes'] = comps.minutes
        
        df['weekend'] = df['date'].dt.dayofweek >= 5
        
        df['coucher_dt'] = df['coucher'] # Garde une copie des datetimes si besoin ailleurs
        df['lever_dt'] = df['lever']
    
        df['mins_coucher'] = df['coucher_dt'].dt.hour * 60 + df['coucher_dt'].dt.minute
        df['mins_lever'] = df['lever_dt'].dt.hour * 60 + df['lever_dt'].dt.minute
        
        # Validation des colonnes requises
        missing = set(COLS_REQ_STATS) - set(df.columns)
        
        if missing:
            logger.error("Colonnes manquantes après nettoyage : %s", missing)
            raise KeyError(f"Colonnes manquantes : {missing}")

        logger.info(
            "prepare_sleep_df terminée : %d lignes, %d colonnes",
            df.shape[0], df.shape[1]
        )
        return df
    except KeyError : raise
    except Exception : logger.exception('Erreur inattendue dans prepare_sleep_df pour %s', path)


def main() :
    
    if not SLEEP_DATA_FILE.exists():
        logger.critical("fichier introuvable : %s  " ,  SLEEP_DATA_FILE )
        raise FileNotFoundError(f"Fichier introuvable : {SLEEP_DATA_FILE}")
    else :
        df = prepare_sleep_df(SLEEP_DATA_FILE)

    stats_glob   = calculer_stats_globales(df)
    stats_periode = calculer_stats_par_periode(df)
    heures_med   = calculer_heures_medianes(df)

    stats_sommeil  = stats_glob + stats_periode + heures_med
    afficher_rapport(stats_sommeil ,df['date'].min().strftime('%A %d %B %Y'),df['date'].max().strftime('%A %d %B %Y'))
#affichage du dataframe
    df_display = df.copy() # Filtre sur une copie
    df_display['jour'] = df_display['date'].dt.day_name()
# Crée une colonne formatée pour l'affichage SEULEMENT dans cette copie
    df_display['duree_h'] = "~" + (df_display['duree'] / 60).round(1).astype(str) + " h"

    df_display =df_display[["date","coucher","lever","jour","duree_h","qualite"]]
    print(df_display )
    visualiser_distribution_duree(df) # df ici est le DataFrame retourné par prepare_sleep_df
    visualiser_evolution_duree(df)
    visualiser_coucher_vs_duree(df)
if __name__ == "__main__" :
    main()

