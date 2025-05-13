from pathlib import Path
import pandas as pd
from datetime import timedelta
from dateutil import parser
from pandas import Timestamp
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt 
from datetime import datetime
import seaborn as sns
import argparse
import os

from reportlab.pdfgen import canvas 



BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
SLEEP_DATA_FILE = DATA_DIR / "in" / "Sommeil.csv"
REPORT_DATA_FILE_PDF = DATA_DIR /"out" / "DataPulse_Sleep_Report.pdf"
REPORT_DATA_FILE_TXT = DATA_DIR /"out" / "DataPulse_Sleep_Report.txt"
IMG_DIR = Path(__file__).resolve().parent.parent / "data" / "out"
IMG_DIST = IMG_DIR / "distribution_duree_sommeil.png"
IMG_EVOL = IMG_DIR / "evolution_duree_sommeil.png"
IMG_COUCH = IMG_DIR / "sommeil_heure_coucher.png"

COLS_REQ_STATS = ['date','coucher','lever','qualite','raw_coucher','raw_lever','only_coucher','only_lever','time_lever','duree','heures','minutes','weekend']

def parse_args():
    parser = argparse.ArgumentParser(description="Analyse de sommeil avec DataPulse")
    parser.add_argument("--file", type=str, default=str(SLEEP_DATA_FILE), help="Chemin vers le fichier CSV d'entrée")
    parser.add_argument("--format", choices=["txt", "pdf", "both"], default="both", help="Format de sortie du rapport")
    parser.add_argument("--viz", action="store_true", help="Afficher et sauvegarder les visualisations")
    return parser.parse_args()

def traduire_date_en_francais(texte: str) -> str:
    """Traduit les jours et mois anglais vers le français dans une chaîne de date.

    Parameters
    ----------
    texte : str
        Chaîne contenant une ou plusieurs dates en anglais.

    Returns
    -------
    str
        Chaîne avec les jours et mois traduits en français.
    """
    jours = {
        "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
        "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"
    }

    mois = {
        "January": "Janvier", "February": "Février", "March": "Mars", "April": "Avril",
        "May": "Mai", "June": "Juin", "July": "Juillet", "August": "Août",
        "September": "Septembre", "October": "Octobre", "November": "Novembre", "December": "Décembre"
    }

    for en, fr in jours.items():
        texte = texte.replace(en, fr)
    for en, fr in mois.items():
        texte = texte.replace(en, fr)
    return texte

def horodater (file_param) : 
    """Écrit dans un fichier texte les données du rapport.

    Parameters
    ----------
    file_param : str
        Nom et emplacement du fichier texte à horodater.
     Returns
     -------
     matplotlib.axes.Axes
         Nom et emplacement du fichier horodaté.
    """
    return (str(file_param).split('.')[0] + '_' + datetime.now().strftime("%Y%m%d%H%M") + '.' + str(file_param).split('.')[1] )
    
     

def ecrire_rapport_pdf(file,texte,marge_gauche=50,marge_haut=800,police='Helvetica', taille_police=8,interligne=10) : 
   """Écrit un rapport PDF avec texte et images, en gérant les erreurs de chargement d'image."""
   c = canvas.Canvas(horodater(file))
   c.setFont(police, taille_police)

   for ligne in texte.split('\n'):
        c.drawString(marge_gauche, marge_haut, ligne)
        marge_haut -= interligne

    # Images (chacune protégée)
   images = [
        (IMG_DIST, 420),
        (IMG_EVOL, 220),
        (IMG_COUCH, 10)
    ]

   for img_path, y in images:
        try:
            if Path(IMG_DIST).exists():
                c.drawImage(str(IMG_DIST), x=50, y=420, width=500, height=200)
            if Path(IMG_EVOL).exists():
                c.drawImage(str(IMG_EVOL), x=50, y=220, width=500, height=200)
            if Path(IMG_COUCH).exists():
                c.drawImage(str(IMG_COUCH), x=50, y=10, width=500, height=200)
        except Exception as e:
            logger.warning("Image introuvable ou non lisible (%s) : %s", img_path, e)

   c.save()
   # Supprimer les images
   for img_path, _ in images:
        try:
            if img_path.exists():
                os.remove(img_path)
                logger.info(f"Image supprimée : {img_path}")
        except Exception as e:
            logger.warning(f"Erreur lors de la suppression de {img_path} : {e}")

def ecrire_rapport_txt(file: str, chaine: str):
    """Écrit dans un fichier texte les données du rapport.

    Parameters
    ----------
    file : str
        Nom et emplacement du fichier texte à créer.
    chaine : str
        Contenu à écrire dans le fichier.
    """
    with open(horodater(file), "w", encoding="utf-8") as f:
        f.write(chaine)

def visualiser_coucher_vs_duree(df: pd.DataFrame, *, avec_regression: bool = True, ax=None):
    """Visualise la durée du sommeil en fonction de l'heure de coucher.

    Affiche un nuage de points de la durée du sommeil contre l'heure de coucher, éventuellement
    accompagné d'une droite de régression linéaire.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les colonnes 'mins_coucher' et 'duree'.
    avec_regression : bool, optional
        Si True, ajoute une droite de régression linéaire au graphique. Par défaut à True.
    ax : matplotlib.axes.Axes, optional
        Objet Axes existant pour tracer le graphique. Si None, un nouveau graphique est créé.

    Returns
    -------
    matplotlib.axes.Axes
        Axes contenant la visualisation créée.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'mins_coucher': [1320, 1410, 1380],
    ...     'duree': [420, 390, 450]
    ... })
    >>> visualiser_coucher_vs_duree(data)
    """
    # Vérifier et nettoyer les données nécessaires
    data_plot = df[['mins_coucher', 'duree']].dropna()
    if data_plot.empty:
        logger.info("Données insuffisantes pour visualiser l'heure de coucher vs durée.")
        return

    # Création des Axes si aucun n'est fourni
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Tracé du nuage de points
    sns.scatterplot(
        data=data_plot,
        x='mins_coucher',
        y='duree',
        alpha=0.6,
        edgecolor="w",
        ax=ax
    )

    # Ajout d'une droite de régression linéaire
    if avec_regression:
        sns.regplot(
            data=data_plot,
            x='mins_coucher',
            y='duree',
            scatter=False,
            ax=ax,
            line_kws={'linewidth': 2, 'color': 'red'}
        )

    # Personnalisation des axes et du titre
    ax.set(
        title="Durée du sommeil en fonction de l'heure de coucher",
        xlabel="Heure de coucher (minutes depuis minuit)",
        ylabel="Durée du sommeil (minutes)"
    )

    # Ajout d'une grille
    ax.grid(True)

    plt.tight_layout()

    # Affichage
    try:
        plt.savefig(IMG_COUCH)
        plt.close()
        logger.info("Visualisation durée du sommeil en fonction de l'heure de coucher créee avec succès.")
    except Exception as e:
        logger.error("Échec de l'affichage : %s", e)

    return ax

     
    
def visualiser_evolution_duree(
    df: pd.DataFrame, *, window: int = 7, ax: plt.Axes = None
):
    """Affiche la durée de sommeil et sa moyenne mobile centrée.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant obligatoirement les colonnes 'date' et 'duree'.
    window : int, default=7
        Taille de la fenêtre glissante (en jours).
    ax : matplotlib.axes.Axes, optional
        Axe matplotlib existant sur lequel tracer. Si None, une nouvelle figure
        sera créée.

    Returns
    -------
    matplotlib.axes.Axes
        L'axe contenant le graphique tracé.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2023-01-01', periods=10),
    ...     'duree': [420, 400, 380, None, 390, 415, 430, 410, 400, 395]
    ... })
    >>> visualiser_evolution_duree(df)

    """
    df_ordonne = (
        df[['date', 'duree']]
        .dropna()
        .sort_values(by='date')
        .reset_index(drop=True)
    )

    if df_ordonne.empty:
        logger.info("Aucune donnée valide pour visualiser l'évolution.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        df_ordonne['date'],
        df_ordonne['duree'],
        marker='o',
        linestyle='-',
        alpha=0.4,
        label='Durée du sommeil',
    )

    moyenne_mobile = (
        df_ordonne.set_index('date')['duree']
        .rolling(window=window, center=True, min_periods=window // 2)
        .mean()
    )

    moyenne_mobile.plot(
        ax=ax,
        linewidth=2,
        label=f"Moyenne mobile ({window} j)",
        color='tab:blue',
    )

    ax.set_title("Évolution de la durée de sommeil", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Durée (minutes)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    try:
        plt.savefig(IMG_EVOL)
        plt.close()
    except Exception as e:
        logger.error("Erreur lors de la visualisation: %s", e)

    logger.info("Visualisation : Évolution de la durée de sommeil créée avec succès.")

    return ax

       
    
def visualiser_distribution_duree(df: pd.DataFrame):
    """Affiche la distribution de la durée du sommeil sous forme d'histogramme.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant une colonne 'duree' représentant les durées de sommeil.

    Returns
    -------
    None

    Examples
    --------
    >>> df = pd.DataFrame({'duree': [420, 400, 380, None, 390, 415, 430, 410, 400, 395]})
    >>> visualiser_distribution_duree(df)
    """
    duree_valide = df['duree'].dropna()

    if duree_valide.empty:
        logger.info("Aucune donnée valide pour générer la visualisation.")
        return

    plt.figure(figsize=(12, 8))
    sns.histplot(duree_valide, kde=True, bins=10, color='skyblue', edgecolor='black')

    plt.title('Distribution de la durée du sommeil', fontsize=16)
    plt.xlabel('Durée du sommeil (minutes)', fontsize=14)
    plt.ylabel('Fréquence (nombre de nuits)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    try:
        plt.savefig(IMG_DIST)
        plt.close() 
    except Exception as e:
        logger.error("Erreur lors de la visualisation : %s", e)

    logger.info("Visualisation Distribution de la durée du sommeil créée avec succès.")
    
      


def configure_logging(log_path="mon_script.log"):
    """Configure un logger global avec fichier rotatif et sortie console.

    Parameters
    ----------
    log_path : str
        Chemin du fichier de log.
    """
    root = logging.getLogger()

    # Suppression des anciens handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Handler pour fichier rotatif
    try:
        fh = RotatingFileHandler(
            log_path,
            maxBytes=5_000_000,
            backupCount=5,
            mode="w",
            encoding="utf-8-sig"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        root.addHandler(fh)
    except PermissionError as e:
        logger.error("Pas de droit d'écriture sur le log : %s", e)

    # Handler pour sortie console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(ch)

    root.setLevel(logging.DEBUG)

configure_logging(DATA_DIR/"mon_script.log")
logger = logging.getLogger(__name__)

def controle_validite(row: pd.Series) -> str:
    """Contrôle la validité des données d'une ligne de DataFrame.

    Parameters
    ----------
    row : pd.Series
        Ligne d'un DataFrame contenant les colonnes "Durée", "Heure de coucher" et "Heure de lever".

    Returns
    -------
    str
        Chaîne décrivant la validité des données.
    """
    invalidite = "valide"

    if row["Durée"] == '--' or pd.isna(row["Durée"]):
        invalidite = "-Durée manquante"
    if row["Heure de coucher"] == '--' or pd.isna(row["Heure de coucher"]):
        invalidite += "-Heure de coucher manquante"
    if row["Heure de lever"] == '--' or pd.isna(row["Heure de lever"]):
        invalidite += "-Heure de lever manquante"

    return invalidite


def lire_fichier_pandas(
    chemin_fichier: str,
    encodages=("utf-8", "latin-1", "cp1252"),
    separateurs=(";", ",", "\t", "|", ":")
):
    """Lit un fichier tabulaire en testant plusieurs encodages et séparateurs.

    Parameters
    ----------
    chemin_fichier : str
        Chemin du fichier à lire.
    encodages : tuple of str
        Encodages à tester (par défaut: utf-8, latin-1, cp1252).
    separateurs : tuple of str
        Séparateurs à tester (par défaut: ; , \t | :).

    Returns
    -------
    dict or None
        Dictionnaire avec les clés 'dataframe', 'encodage_utilisé' et 'séparateur_utilisé', ou None en cas d'échec.
    """
    for encodage in encodages:
        for sep in separateurs:
            try:
                df = pd.read_csv(chemin_fichier, encoding=encodage, sep=sep)

                if not df.empty and len(df.columns) > 1:
                    logger.info("Fichier %s lu avec succès", chemin_fichier)
                    return {
                        "dataframe": df,
                        "encodage_utilisé": encodage,
                        "séparateur_utilisé": sep
                    }

            except UnicodeDecodeError as e:
                logger.warning("Erreur d'encodage avec %s : %s", encodage, e)
            except pd.errors.ParserError as e:
                logger.warning("Erreur de séparateur avec %s : %s", sep, e)
            except Exception as e:
                logger.warning("Erreur inattendue avec %s : %s", sep, e)

    logger.critical("Échec : Impossible de lire %s avec les encodages et séparateurs fournis.", chemin_fichier)
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
    """Convertit une chaîne horaire en objet datetime.time grâce à dateutil.

    Parameters
    ----------
    ts : str or float or pd.NaT
        Chaîne représentant une heure (ex. "03:56 AM") ou une valeur manquante.

    Returns
    -------
    datetime.time or None
        L'heure extraite, ou None en cas d’erreur ou de valeur manquante.
    """
    if pd.isna(ts):
        return None

    try:
        return parser.parse(ts).time()
    except Exception as e:
        logger.debug("Impossible de parser %r en time : %s", ts, e)
        return None

    
def combine(row: pd.Series, time_col: str) -> pd.Timestamp:
    """Construit un Timestamp à partir d'une date et d'une heure dans une ligne.

    Parameters
    ----------
    row : pd.Series
        Ligne contenant la date dans 'date' et l'heure dans la colonne `time_col`.
    time_col : str
        Nom de la colonne contenant l'heure (objet datetime.time).

    Returns
    -------
    pd.Timestamp or pd.NaT
        Timestamp combinant date et heure, ou pd.NaT si l'heure est absente.
    """
    t = row[time_col]
    if t is None:
        logger.debug("Aucune heure détectée dans '%s' pour la date %s", time_col, row.get('date'))
        return pd.NaT

    return Timestamp(
        year=row['date'].year,
        month=row['date'].month,
        day=row['date'].day,
        hour=t.hour,
        minute=t.minute
    )

def calculer_stats_globales(df: pd.DataFrame) -> tuple:
    """Calcule les statistiques globales de durée de sommeil.

    Parameters
    ----------
    df : pd.DataFrame
        Doit contenir une colonne 'duree' (en minutes).

    Returns
    -------
    tuple
        (totales, normales, courtes, longues, moyenne, mediane, minimale, maximale, ecart_type)
    """
    logger.info("Début du calcul des stats globales sur %d lignes", len(df))

    df_clean = df[df['duree'].notna()]

    try:
        if df_clean.empty:
            logger.warning("Aucune donnée valide pour le calcul des statistiques")
            return (0, 0, 0, 0, 0, 0, 0, 0, 0)

        totales = len(df_clean)
        normales = df_clean[(df_clean['duree'] >= 420) & (df_clean['duree'] <= 540)].shape[0]
        courtes = df_clean[df_clean['duree'] < 420].shape[0]
        longues = df_clean[df_clean['duree'] > 540].shape[0]
        moyenne = df_clean['duree'].mean()
        mediane = df_clean['duree'].median()
        minimale = df_clean['duree'].min()
        maximale = df_clean['duree'].max()
        ecart_type = df_clean['duree'].std()

        logger.info("Stats globales — totales: %d, moy: %.1f, médiane: %.1f, min: %.1f, max: %.1f", totales, moyenne, mediane, minimale, maximale)

        return totales, normales, courtes, longues, moyenne, mediane, minimale, maximale, ecart_type

    except Exception as e:
        logger.exception("Erreur lors du calcul des stats globales : %s", e)
        return (0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

def calculer_stats_par_periode(df: pd.DataFrame):
    """Retourne les moyennes de durée de sommeil pour semaine et week-end.

    Parameters
    ----------
    df : pd.DataFrame
        Doit contenir une colonne 'date' (datetime) et 'duree'.

    Returns
    -------
    tuple
        (moyenne_semaine, moyenne_weekend)
    """
    logger.info("Début du calcul des stats par période")

    try:
        moyennes = df.groupby(df['date'].dt.weekday >= 5)['duree'].mean()
        moyennes = moyennes.reindex([False, True], fill_value=0)

        logger.info("Stats périodiques — weekend: %.1f, semaine: %.1f", moyennes[True], moyennes[False])

        return moyennes[False], moyennes[True]

    except Exception as e:
        logger.exception("Erreur lors du calcul des stats par période : %s", e)
        return (0.0, 0.0)

def calculer_heures_medianes(df: pd.DataFrame) -> tuple:
    """Calcule les heures médianes de coucher et de lever en minutes.

    Parameters
    ----------
    df : pd.DataFrame
        Doit contenir les colonnes 'mins_coucher' et 'mins_lever'.

    Returns
    -------
    tuple
        (median_coucher, median_lever)
    """
    logger.info("Début du calcul des heures médianes")

    try:
        median_coucher = df['mins_coucher'].dropna().median()
        median_lever = df['mins_lever'].dropna().median()

        logger.info("Heures médianes calculées — coucher: %d min, lever: %d min", median_coucher, median_lever)
        return median_coucher, median_lever

    except Exception as e:
        logger.exception("Erreur lors du calcul des heures médianes : %s", e)
        return 0.0, 0.0
        


def creer_rapport(stats_sommeil: tuple, date_min, date_max) -> None:
    """Affiche un rapport synthétique de sommeil sur la période spécifiée.

    Parameters
    ----------
    stats_sommeil : tuple
        (totales, normales, courtes, longues, moyenne, mediane, minimale,
         maximale, ecart_type, moy_semaine, moy_weekend, heure_median_coucher, heure_median_lever)
    date_min : str or pd.Timestamp
        Date de début de la période.
    date_max : str or pd.Timestamp
        Date de fin de la période.

    Returns
    -------
    None
    """
    chaine  = traduire_date_en_francais (f"DataPulse - Rapport Sommeil du {date_min} au {date_max}\n")
    chaine +=f"\nNombre de nuits analysées   : {stats_sommeil[0]}\n"
    chaine +=f"  - normales                            : {stats_sommeil[1]}\n"
    chaine +=f"  - courtes                               : {stats_sommeil[2]}\n"
    chaine +=f"  - longues                              : {stats_sommeil[3]}\n"
    chaine +=f"Durée moyenne                     : {stats_sommeil[4]:.0f} min (~{stats_sommeil[4]/60:.1f} h)\n"
    chaine +=f"Durée médiane(1)                  : {stats_sommeil[5]:.0f} min (~{stats_sommeil[5]/60:.1f} h)\n"
    chaine +=f"Durée minimale                      : {stats_sommeil[6]:.0f} min (~{stats_sommeil[6]/60:.1f} h)\n"
    chaine +=f"Durée maximale                     : {stats_sommeil[7]:.0f} min (~{stats_sommeil[7]/60:.1f} h)\n"
    chaine +=f"Écart-type                               : {stats_sommeil[8]:.0f} min\n"
    chaine +=f"Durée moyenne en semaine   : {stats_sommeil[9]:.0f} min (~{stats_sommeil[9]/60:.1f} h)\n"
    chaine +=f"Durée moyenne le week-end  : {stats_sommeil[10]:.0f} min (~{stats_sommeil[10]/60:.1f} h)\n"
    chaine +=f"Heure médiane de coucher(1) : {mins_to_hhmm(stats_sommeil[11])}\n"
    chaine +=f"Heure médiane de lever(1)      : {mins_to_hhmm(stats_sommeil[12])}\n"
    chaine +="\n--------------------------------------------------------------------------------------------------------------\n"
    chaine +="\n(1) Médiane moins sensible aux valeurs extrêmes"
    chaine +="\n(2) Écart-type faible → sommeil régulier"
    return chaine

    


def prepare_sleep_df(path: str) -> pd.DataFrame | None:
    """Charge et prépare un DataFrame de sommeil depuis un fichier CSV.

    Parameters
    ----------
    path : str or pathlib.Path
        Chemin vers le fichier CSV de sommeil.

    Returns
    -------
    pd.DataFrame or None
        Le DataFrame enrichi des colonnes :
        ['date', 'coucher', 'lever', 'duree', 'heures', 'minutes', …]
    """
    try:
        logger.info("Début de prepare_sleep_df pour %s", path)

        resultat = lire_fichier_pandas(path)
        if resultat is None:
            logger.critical("Abandon de prepare_sleep_df : échec de lecture de %s", path)
            return None

        df = resultat["dataframe"]
        df.columns = df.columns.str.replace('\u00A0', ' ', regex=False)

        df["qualite"] = df.apply(controle_validite, axis=1)
        logger.debug("Avant drop/rename : shape=%s, colonnes=%s", df.shape, df.columns.tolist())

        df = (
            df
            .drop(columns=['Durée'], errors='ignore')
            .rename(columns={
                'Sommeil 4 semaines': 'date',
                'Heure de coucher': 'coucher',
                'Heure de lever': 'lever'
            })
        )
        logger.debug("Après drop/rename  : shape=%s, colonnes=%s", df.shape, df.columns.tolist())

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().any():
            logger.warning("Certaines dates n'ont pas pu être parsées")
        if df['date'].isna().all():
            logger.warning("Toutes les dates sont NaT après conversion pour %s", path)

        df['raw_coucher'] = df['coucher'].astype(str).replace('--', pd.NA)
        df['raw_lever'] = df['lever'].astype(str).replace('--', pd.NA)

        for col in ['raw_coucher', 'raw_lever']:
            df[col] = (
                df[col]
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
                .str.strip("'")
                .str.replace(r'^(\d):', r'0\1:', regex=True)
            )

        time_pattern = r'(\d{2}:\d{2}\s?[AP]M)'
        df['only_coucher'] = df['raw_coucher'].str.extract(time_pattern, expand=False).str.strip()
        df['only_lever'] = df['raw_lever'].str.extract(time_pattern, expand=False).str.strip()

        df['time_coucher'] = df['only_coucher'].apply(parse_time_ts)
        df['time_lever'] = df['only_lever'].apply(parse_time_ts)

        df['coucher'] = df.apply(combine, args=('time_coucher',), axis=1)
        df['lever'] = df.apply(combine, args=('time_lever',), axis=1)

        mask = df['lever'] <= df['coucher']
        df.loc[mask, 'lever'] += pd.Timedelta(days=1)

        df['duree'] = (df['lever'] - df['coucher']).dt.total_seconds() / 60
        comps = (df['lever'] - df['coucher']).dt.components
        df['heures'] = comps.days * 24 + comps.hours
        df['minutes'] = comps.minutes

        df['weekend'] = df['date'].dt.dayofweek >= 5

        df['coucher_dt'] = df['coucher']
        df['lever_dt'] = df['lever']
        df['mins_coucher'] = df['coucher_dt'].dt.hour * 60 + df['coucher_dt'].dt.minute
        df['mins_lever'] = df['lever_dt'].dt.hour * 60 + df['lever_dt'].dt.minute

        missing = set(COLS_REQ_STATS) - set(df.columns)
        if missing:
            logger.error("Colonnes manquantes après nettoyage : %s", missing)
            raise KeyError(f"Colonnes manquantes : {missing}")

        logger.info("prepare_sleep_df terminée : %d lignes, %d colonnes", df.shape[0], df.shape[1])
        return df
    except KeyError:
        raise
    except Exception:
        logger.exception("Erreur inattendue dans prepare_sleep_df pour %s", path)
        return None
    
def main() -> None:
    """Fonction principale exécutant l'analyse et le rapport de sommeil."""
    args = parse_args()
    input_path = Path(args.file)
    # if not SLEEP_DATA_FILE.exists():
    if not input_path.exists():
        logger.critical("Fichier introuvable : %s", input_path)
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    df = prepare_sleep_df(input_path)
    if df is None:
        return

    stats_glob = calculer_stats_globales(df)
    stats_periode = calculer_stats_par_periode(df)
    heures_med = calculer_heures_medianes(df)

    stats_sommeil = stats_glob + stats_periode + heures_med

    chaine = creer_rapport(
        stats_sommeil,
        df['date'].min().strftime('%A %d %B %Y'),
        df['date'].max().strftime('%A %d %B %Y')
    )
    
    print(chaine)
    
    df_display = df.copy()
    df_display['jour'] = df_display['date'].dt.day_name()
    df_display['duree_h'] = "~" + (df_display['duree'] / 60).round(1).astype(str) + " h"
    df_display = df_display[["date", "coucher", "lever", "jour", "duree_h", "qualite"]]
    
    print(df_display)
    
    
    if args.viz:
        visualiser_distribution_duree(df)
        visualiser_evolution_duree(df)
        visualiser_coucher_vs_duree(df)
        
    if args.format in ["pdf", "both"]:
        ecrire_rapport_pdf(REPORT_DATA_FILE_PDF, chaine)
    if args.format in ["txt", "both"]:
        ecrire_rapport_txt(REPORT_DATA_FILE_TXT, chaine)



if __name__ == "__main__":
    main()
