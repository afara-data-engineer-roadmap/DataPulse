import argparse
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dateutil import parser
from pandas import Timestamp
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SLEEP_DATA_FILE = DATA_DIR / "in" / "Sommeil.csv"
REPORT_DATA_FILE_PDF = DATA_DIR / "out" / "DataPulse_Sleep_Report.pdf"
REPORT_DATA_FILE_TXT = DATA_DIR / "out" / "DataPulse_Sleep_Report.txt"
IMG_DIR = Path(__file__).resolve().parent.parent / "data" / "out"
IMG_DIST = IMG_DIR / "distribution_duree_sommeil.png"
IMG_EVOL = IMG_DIR / "evolution_duree_sommeil.png"
IMG_COUCH = IMG_DIR / "sommeil_heure_coucher.png"
COLS_REQ_STATS = [
    "date",
    "coucher",
    "lever",
    "qualite",
    "raw_coucher",
    "raw_lever",
    "only_coucher",
    "only_lever",
    "time_lever",
    "duree",
    "heures",
    "minutes",
    "weekend",
]


def configure_logging(log_path="data_pulse_log.log"):
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

    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler pour fichier rotatif
    try:
        fh = RotatingFileHandler(
            log_path,
            maxBytes=5_000_000,
            backupCount=5,
            mode="w",
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_formatter)
        root.addHandler(fh)
    except PermissionError as e:
        logger.error("Pas de droit d'écriture sur le log : %s", e)

    # Handler pour sortie console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(log_formatter)
    root.addHandler(ch)
    root.setLevel(logging.DEBUG)


configure_logging(DATA_DIR / "mon_script.log")
logger = logging.getLogger(__name__)


def parse_args():
    """
    Analyse et retourne les arguments de la ligne de commande.

    Cette fonction configure et utilise argparse pour permettre à l'utilisateur
    de spécifier des options lors de l'exécution du script, telles que
    le fichier d'entrée, le format de sortie du rapport, et si les
    visualisations doivent être affichées et sauvegardées.

    Returns
    -------
    argparse.Namespace
        Un objet contenant les arguments parsés. Les arguments sont accessibles
        comme des attributs de cet objet (par exemple, `args.file`,
                                          `args.format`,`args.viz`).
        - file (str): Chemin vers le fichier CSV d'entrée.
                      Par défaut : la valeur de la constante SLEEP_DATA_FILE.
        - format (str): Format de sortie du rapport. Choix possibles : "txt",
        "pdf", "both".
                        Par défaut : "both".
        - viz (bool): Si True, affiche et sauvegarde les visualisations.
                      Par défaut : False (l'action "store_true" met à True
                                          si l'option est présente).
    """
    parser = argparse.ArgumentParser(description="Analyse  sommeil DataPulse")
    parser.add_argument(
        "--file",
        type=str,
        default=str(
            SLEEP_DATA_FILE
        ),  # s'assurer que SLEEP_DATA_FILE est un str ou Path
        help="Chemin vers le fichier CSV d'entrée des données de sommeil.",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "pdf", "both"],
        default="both",
        help="Format de sortie du rapport ('txt', 'pdf', ou 'both').",
    )
    parser.add_argument(
        "--viz",  # L'argumente est True si --viz est présent, False sinon
        action="store_true",
        help="Afficher et sauvegarder les visualisations graphiques.",
    )
    return parser.parse_args()


def formater_serie_dates_fr(serie_dt: pd.Series, format_str="%A %d %B %Y"):
    """Traduit une série de dates datetime en format français
    Parameters
    ----------
    serie_datetime : pd.Series
        Colonne de type datetime.
    format_str : str
        Format de date à appliquer avant traduction.
    Returns
    -------
    pd.Series
        Chaîne de dates traduites en français.
    """
    jours = {
        "Monday": "Lundi",
        "Tuesday": "Mardi",
        "Wednesday": "Mercredi",
        "Thursday": "Jeudi",
        "Friday": "Vendredi",
        "Saturday": "Samedi",
        "Sunday": "Dimanche",
    }

    mois = {
        "January": "Janvier",
        "February": "Février",
        "March": "Mars",
        "April": "Avril",
        "May": "Mai",
        "June": "Juin",
        "July": "Juillet",
        "August": "Août",
        "September": "Septembre",
        "October": "Octobre",
        "November": "Novembre",
        "December": "Décembre",
    }
    try:
        dates_str = serie_dt.dt.strftime(format_str)
        dates_str = dates_str.replace(jours, regex=True)
        dates_str = dates_str.replace(mois, regex=True)
        logger.info("[FORMAT_FR] Formatage des dates  effectué avec succès")
    except Exception as e:
        logger.warning(
            "[FORMAT_FR] Erreur lors du formatage des dates en français : %s",
            e,
        )
    return dates_str


def traduire_texte_date_en_francais(texte: str):
    """
    Traduit les jours et mois anglais vers le français dans une chaîne de date.

    Parameters
    ----------
    texte : str
        Chaîne contenant une ou plusieurs dates en anglais.

    Returns
    -------
    str
        Chaîne avec les jours et mois traduits en français.
    """
    if not isinstance(texte, str):
        logger.warning(
            "[TRAD_DATE] Entrée non valide, texte attendu mais reçu : %s",
            type(texte),
        )
        return texte  # ou raise TypeError si tu veux l'interrompre

    jours = {
        "Monday": "Lundi",
        "Tuesday": "Mardi",
        "Wednesday": "Mercredi",
        "Thursday": "Jeudi",
        "Friday": "Vendredi",
        "Saturday": "Samedi",
        "Sunday": "Dimanche",
    }

    mois = {
        "January": "Janvier",
        "February": "Février",
        "March": "Mars",
        "April": "Avril",
        "May": "Mai",
        "June": "Juin",
        "July": "Juillet",
        "August": "Août",
        "September": "Septembre",
        "October": "Octobre",
        "November": "Novembre",
        "December": "Décembre",
    }

    for en, fr in jours.items():
        texte = texte.replace(en, fr)
    for en, fr in mois.items():
        texte = texte.replace(en, fr)

    logger.info("[TRAD_DATE] Traduction effectuée : %s", texte)
    return texte


def horodater(file_param: str):
    """
    Génère un nom de fichier horodaté basé sur un fichier fourni.

    Parameters
    ----------
    file_param : str
        Nom et chemin du fichier de base (ex: "rapport.txt").

    Returns
    -------
    str
        Nom du fichier avec horodatage (ex: "rapport_20250516_1957.txt").
    """
    try:
        path = Path(file_param)
        horodatage = datetime.now().strftime("%Y%m%d_%H%M")
        nouveau_nom = f"{path.stem}_{horodatage}{path.suffix}"
        nouveau_chemin = path.with_name(nouveau_nom)
        logger.info("[HOROD] Fichier horodaté généré : %s", nouveau_chemin)
        return str(nouveau_chemin)  # Retourne str pour compatibilité/docstring
    except Exception as e:
        logger.warning(
            "[HOROD] Erreur lors de la génération du fichier horodaté : %s",
            e,
        )
        return file_param  # En cas d'erreur, on retourne le nom original


def ecrire_rapport_pdf(
    fichier_pdf: str | Path,
    texte: str,
    marge_gauche: int = 50,
    marge_haut: int = 800,
    police: str = "Helvetica",
    taille_police: int = 8,
    interligne: int = 10,
):
    """
    Génère un rapport PDF contenant un texte multi-ligne et, le cas échéant,
    jusqu’à trois images empilées.

    Parameters
    ----------
    fichier_pdf : str | Path
        Nom (ou chemin) cible du PDF *avant* horodatage.
    texte : str
        Contenu textuel (plusieurs lignes séparées par '\n').
    marge_gauche, marge_haut : int
        Coordonnées initiales en points (72 pt = 2,54 cm).
    police : str
        Police ReportLab (par ex. 'Helvetica', 'Times-Roman'…).
    taille_police : int
        Corps de la police.
    interligne : int
        Pas vertical entre deux lignes de texte, en points.
    images : list[(Path, y)]
        Liste (chemin, ordonnée) pour chaque image.
        Si None, on prend les constantes globales IMG_DIST/EVOL/COUCH.

    Returns
    -------
    Path
        Chemin complet du PDF horodaté généré.
    """
    # ------------------------------------------------------------------ #
    # 0. Préparation des images par défaut si paramètre non fourni
    # ------------------------------------------------------------------ #

    images = [
        (Path(IMG_DIST), 420),
        (Path(IMG_EVOL), 220),
        (Path(IMG_COUCH), 10),
    ]

    # ------------------------------------------------------------------ #
    # 1. Création du PDF
    # ------------------------------------------------------------------ #
    output_path: Path = horodater(fichier_pdf)
    c = canvas.Canvas(str(output_path), pagesize=A4)
    c.setFont(police, taille_police)

    # ------------------------------------------------------------------ #
    # 2. Écriture du texte ligne par ligne
    # ------------------------------------------------------------------ #
    y = marge_haut
    for ligne in texte.splitlines():
        c.drawString(marge_gauche, y, ligne)
        y -= interligne
        if y < 50:  # rudimentaire : nouvelle page si bas
            c.showPage()
            c.setFont(police, taille_police)
            y = marge_haut

    # ------------------------------------------------------------------ #
    # 3. Insertion d’images (protégée)
    # ------------------------------------------------------------------ #
    for img, y_img in images:
        try:
            if img.exists():
                c.drawImage(str(img), x=50, y=y_img, width=500, height=200)
                logger.debug("[EXPORT] Image insérée : %s", img)
            else:
                logger.warning("[EXPORT] Image manquante : %s", img)
        except Exception as e:
            logger.warning("[EXPORT] Problème à lʼinsertion de %s: %s", img, e)

    c.save()
    logger.info("[EXPORT] PDF généré avec succès : %s", output_path)

    # ------------------------------------------------------------------ #
    # 4. Nettoyage optionnel des fichiers image
    # ------------------------------------------------------------------ #
    for img_path, _ in images:
        try:
            if img_path.exists():
                img_path.unlink()
                logger.info("[EXPORT] Image temporaire supprimée : %s", img)
        except Exception as e:
            logger.warning("[EXPORT] Échec de suppression de %s : %s", img, e)

    return output_path


def ecrire_rapport_txt(file: str, contenu_rapport: str):
    """Écrit dans un fichier texte les données du rapport.

    Parameters
    ----------
    file : str
        Nom et emplacement du fichier texte à créer.
    contenu_rapport  : str
        Contenu à écrire dans le fichier.
    """
    with open(horodater(file), "w", encoding="utf-8") as f:
        f.write(contenu_rapport)


def visualiser_coucher_vs_duree(
    df: pd.DataFrame, *, avec_regression: bool = True, ax=None
):
    """Visualise la durée du sommeil en fonction de l'heure de coucher.

    Affiche un nuage de points de la durée du sommeil contre l'heure de coucher
    , éventuellement
    accompagné d'une droite de régression linéaire.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les colonnes 'mins_coucher' et 'duree'.
    avec_regression : bool, optional
        Si True, ajoute une droite de régression linéaire au graphique.
        Par défaut à True.
    ax : matplotlib.axes.Axes, optional
        Objet Axes existant pour tracer le graphique. Si None,
        un nouveau graphique est créé.

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
    data = df[["mins_coucher", "duree"]].dropna()
    if data.empty:
        logger.info("[VISUAL] Données insuff. pr visualiser heure de coucher")
        return

    # Création des axes si aucun n'est fourni
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Tracé du nuage de points
    sns.scatterplot(
        data=data,
        x="mins_coucher",
        y="duree",
        alpha=0.6,  # Transparence des points
        edgecolor="w",  # Couleur du contour des points
        ax=ax,  # Axes sur lesquels dessiner
    )

    # Ajout d'une droite de régression linéaire
    if avec_regression:
        sns.regplot(
            data=data,
            x="mins_coucher",
            y="duree",
            scatter=False,
            ax=ax,
            line_kws={
                "linewidth": 2,
                "color": "red",
            },  # Style ligne régression
            # 'linewidth': épaisseur
        )

    # Personnalisation des axes et du titre
    ax.set(
        title="Durée du sommeil en fonction de l'heure de coucher",
        xlabel="Heure de coucher (minutes depuis minuit)",
        ylabel="Durée du sommeil (minutes)",
        grid=True,  # Ajout d'une grille
    )

    plt.tight_layout()  # Ajuste auto. les élémts pr éviter les superpositions

    # sauvegarde
    try:
        plt.savefig(IMG_COUCH)
        plt.close()
        logger.info(
            "[VISUAL] Visualisation durée du sommeil en fonction de "
            + "l'heure de coucher créee avec succès."
        )
    except Exception as e:
        logger.error("[VISUAL] Échec de l'affichage : %s", e)

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
    # Vérifier et nettoyer les données nécessaires
    data = df[["date", "duree"]].dropna()
    data = data.datasort_values(by="date")
    data = data.reset_index(drop=True)

    if data.empty:
        logger.info("[VISUAL] Aucune donnée valide pour  évolution.")
        return
    # Création des Axes si aucun n'est fourni
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Tracé  courbe
    ax.plot(
        data["date"],
        data["duree"],
        marker="o",
        linestyle="-",
        alpha=0.4,
        label="Durée du sommeil",
    )

    moyenne_mobile = (
        data.set_index("date")["duree"]  # Index par date pr rolling() temporel
        .rolling(
            window=window, center=True, min_periods=window // 2
        )  # Fenêtre glissante
        .mean()
    )

    # Tracé  nuage de points
    moyenne_mobile.plot(
        ax=ax,
        linewidth=2,
        label=f"Moyenne mobile ({window} j)",  # Nom pour la légende
        color="tab:blue",
    )

    # Personnalisation de l'axe et du titre
    ax.set(
        title="Évolution de la durée de sommeil",
        xlabel="Date",
        ylabel="Durée (minutes)",
        grid=True,
        linestyle="--",
        alpha=0.6,
    )
    ax.legend()  # Afficher la légende pour distinguer les deux lignes

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Affichage
    try:
        plt.savefig(IMG_EVOL)
        plt.close()
        logger.info("[VISUAL] Évolution durée de sommeil créee avec succès.")
    except Exception as e:
        logger.error("[VISUAL] Erreur lors de la visualisation: %s", e)

    return ax


def visualiser_distribution_duree(df: pd.DataFrame):
    """Affiche la distribution de la durée du sommeil sous forme d'histogramme.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant une colonne 'duree' représentant les durées de
        sommeil.

    Returns
    -------
    None

    Examples
    --------
    >>> df = pd.DataFrame({'duree':
                          [420, 400, 380, None, 390, 415, 430, 410, 400, 395]})
    >>> visualiser_distribution_duree(df)
    """
    # Vérifier et nettoyer les données nécessaires
    data = df["duree"].dropna()

    if data.empty:
        logger.info("Aucune donnée valide pour générer la visualisation.")
        return
    # Personnalisation
    plt.figure(figsize=(12, 8))
    sns.histplot(data, kde=True, bins=10, color="skyblue", edgecolor="black")

    plt.set(
        title="Distribution de la durée du sommeil",
        xlabel="Durée du sommeil (minutes)",
        ylabel="Fréquence (nombre de nuits)",
        grid=True,
    )
    # Pas de légende nécessaire ici car une seule distribution est tracée

    # sauvegarde
    try:
        plt.savefig(IMG_DIST)
        plt.close()
    except Exception as e:
        logger.error("Erreur lors de la visualisation : %s", e)

    logger.info("[VISUAL] Distribution durée du sommeil créée avec succès.")


def controle_validite(row: pd.Series) -> str:
    """
    Contrôle la validité d'une ligne de données sommeil.

    Retourne "valide" si tout est ok,
    sinon une chaîne listant les champs manquants.
    """
    erreurs = []

    if row.get("Durée", "--") == "--" or pd.isna(row.get("Durée")):
        erreurs.append("Durée manquante")
    if row.get("Heure de coucher", "--") == "--" or pd.isna(
        row.get("Heure de coucher")
    ):
        erreurs.append("Heure de coucher manquante")
    if row.get("Heure de lever", "--") == "--" or pd.isna(row.get("Heure de lever")):
        erreurs.append("Heure de lever manquante")

    if erreurs:
        message = "; ".join(erreurs)
        logger.debug("[PREPARE] Ligne invalide détectée : %s", message)
        return message
    else:
        return "valide"


def lire_fichier_pandas(
    chemin_fichier: str,
    encodages=("utf-8", "latin-1", "cp1252"),
    separateurs=(";", ",", "\t", "|", ":"),
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
    dict ou None
        Dictionnaire avec les clés 'dataframe', 'encodage_utilisé' et
        'séparateur_utilisé', ou None en cas d'échec.
    """
    for encodage in encodages:
        for s in separateurs:
            try:

                if hasattr(
                    chemin_fichier, "seek"
                ):  # Si chemin_fichier est un objet fichier en mémoire
                    chemin_fichier.seek(0)  # On "rembobine" au début du flux
                df = pd.read_csv(chemin_fichier, encoding=encodage, sep=s)

                if (
                    not df.empty and len(df.columns) > 1
                ):  # DataFrame non vide et plus d'une colonne
                    logger.info("Fichier %s lu avec succès", chemin_fichier)
                    return {
                        "dataframe": df,
                        "encodage_utilisé": encodage,
                        "séparateur_utilisé": s,
                    }

            except UnicodeDecodeError as e:
                logger.warning(
                    "[PARSE_CSV] Erreur d'encodage avec %s : %s", encodage, e
                )
            except pd.errors.ParserError as e:
                logger.warning("[PARSE_CSV] Err séparateur avec %s : %s", s, e)
            except Exception as e:
                logger.warning("[PARSE_CSV] Err inattendue avec %s : %s", s, e)

    logger.critical(
        "[PARSE_CSV] Échec : Impossible de lire %s avec les encodages et "
        + "séparateurs fournis.",
        chemin_fichier,
    )
    return None


def mins_to_hhmm(total_minutes):
    """
    Convertit un nombre de minutes depuis minuit en chaîne 'HH:MM'.

    Parameters
    ----------
    total_minutes : float | int | NaN
        Minutes écoulées depuis 00:00. Peut être fractionnaire ou NaN.

    Returns
    -------
    str
        'HH:MM' ou 'N/A' si la valeur est manquante.
    """
    # Valeur manquante → N/A
    if pd.isna(total_minutes):
        return "N/A"

    # Valeur négative ou absurde → log en DEBUG, puis correction modulo 24 h
    if total_minutes < 0:
        logger.debug("[UTIL] mins_to_hhmm : minutes nég (%s)", total_minutes)

    total_minutes = int(total_minutes)
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def parse_time_ts(ts):
    """
    Convertit une chaîne horaire en objet datetime.time grâce à dateutil.

    Parameters
    ----------
    ts : str | float | pd.NaT
        Chaîne représentant une heure (ex. "03:56 AM") ou valeur manquante.

    Returns
    -------
    datetime.time | None
        Heure extraite, ou None en cas d’erreur / valeur manquante.
    """
    if pd.isna(ts):  # Valeur absurde
        return None

    try:
        return parser.parse(ts).time()
    except Exception as e:
        logger.debug("[UTIL] parse_time_ts — imposs. de parser %r : %s", ts, e)
        return None


def combine(row: pd.Series, time_col: str):
    """
    Construit un Timestamp à partir d'une date et d'une heure.

    Parameters
    ----------
    row : pd.Series
        Ligne contenant la date ('date') et l'heure dans `time_col`.
    time_col : str
        Nom de la colonne contenant l'heure (datetime.time).

    Returns
    -------
    pd.Timestamp | pd.NaT
        Timestamp combinant date et heure, ou pd.NaT si l'heure est absente.
    """
    # Récupère l'objet datetime.time (ou None) de la cln spécifiée
    t = row[time_col]
    if t is None:  # Vérifie si l'heure est absente
        logger.debug(
            "[UTIL] combine — heure manquante dans '%s' pour la date %s",
            time_col,
            row.get("date"),
        )
        return pd.NaT

    # on construit l'objet Timestamp
    return Timestamp(
        year=row["date"].year,
        month=row["date"].month,
        day=row["date"].day,
        hour=t.hour,
        minute=t.minute,
    )


def calculer_stats_globales(df: pd.DataFrame):
    """
    Calcule les statistiques globales de durée de sommeil.

    Returns
    -------
    tuple :
        (totales, normales, courtes, longues,
         moyenne, mediane, minimale, maximale, ecart_type)
    """
    logger.info("[STATS_GLOBAL] Début sur %d lignes", len(df))
    df_clean = df[df["duree"].notna()]

    try:
        if df_clean.empty:
            logger.warning("[STATS_GLOBAL] Aucune donnée valide")
            return (0, 0, 0, 0, 0, 0, 0, 0, 0)

        totales = len(df_clean)
        normales = df_clean[
            (df_clean["duree"] >= 420) & (df_clean["duree"] <= 540)
        ].shape[0]
        courtes = df_clean[df_clean["duree"] < 420].shape[0]
        longues = df_clean[df_clean["duree"] > 540].shape[0]
        moyenne = df_clean["duree"].mean()
        mediane = df_clean["duree"].median()
        minimale = df_clean["duree"].min()
        maximale = df_clean["duree"].max()
        ecart_type = df_clean["duree"].std()

        logger.info(
            "[STATS_GLOBAL] totales=%d, moy=%.1f, médiane=%.1f, min=%.1f, "
            + "max=%.1f",
            totales,
            moyenne,
            mediane,
            minimale,
            maximale,
        )
        return (
            totales,
            normales,
            courtes,
            longues,
            moyenne,
            mediane,
            minimale,
            maximale,
            ecart_type,
        )

    except Exception as e:
        logger.exception("[STATS_GLOBAL] Erreur : %s", e)
        return (0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)


def calculer_stats_par_periode(df: pd.DataFrame):
    """
    Retourne les moyennes de durée de sommeil pour semaine et week-end.

    Returns
    -------
    tuple
        (moyenne_semaine, moyenne_weekend)
    """
    logger.info("[STATS_PERIODE] Début")

    try:
        moyennes = df.groupby(df["date"].dt.weekday >= 5)[
            "duree"
        ].mean()  # True  => week-end
        # cree un grp 0 dans le cas ou pas d'entrée True
        moyennes = moyennes.reindex([False, True], fill_value=0)

        logger.info(
            "[STATS_PERIODE] semaine=%.1f, weekend=%.1f",
            moyennes[False],
            moyennes[True],
        )
        return moyennes[False], moyennes[True]

    except Exception as e:
        logger.exception("[STATS_PERIODE] Erreur : %s", e)
        return (0.0, 0.0)


def calculer_heures_medianes(df: pd.DataFrame):
    """
    Calcule les heures médianes de coucher et de lever (en minutes).

    Returns
    -------
    tuple
        (median_coucher, median_lever)
    """
    logger.info("[STATS_MEDIAN] Début")

    try:
        median_coucher = df["mins_coucher"].dropna().median()
        median_lever = df["mins_lever"].dropna().median()

        logger.info(
            "[STATS_MEDIAN] coucher=%d min, lever=%d min",
            median_coucher,
            median_lever,
        )
        return median_coucher, median_lever

    except Exception as e:
        logger.exception("[STATS_MEDIAN] Erreur : %s", e)
        return 0.0, 0.0


def creer_rapport(stats_sommeil: tuple, date_min, date_max):
    """
    Génère le texte du rapport synthétique de sommeil pour la période donnée.
    """
    rapport_txt = traduire_texte_date_en_francais(
        f"DataPulse - Rapport Sommeil du {date_min} au {date_max}\n"
    )
    rapport_txt += f"\nNombre de nuits analysées   : {stats_sommeil[0]}\n"
    rapport_txt += "  - normales                            :"
    rapport_txt += f" {stats_sommeil[1]}\n"
    rapport_txt += "  - courtes                              : "
    rapport_txt += f"{stats_sommeil[2]}\n"
    rapport_txt += "  - longues                              : "
    rapport_txt += f"{stats_sommeil[3]}\n"
    rapport_txt += (
        f"Durée médiane(1)                  : "
        f"{stats_sommeil[5]:.0f} min (~{stats_sommeil[5]/60:.1f} h)\n"
    )
    rapport_txt += (
        f"Durée minimale                      : "
        f"{stats_sommeil[6]:.0f} min (~{stats_sommeil[6]/60:.1f} h)\n"
    )
    rapport_txt += (
        f"Durée maximale                     : "
        f"{stats_sommeil[7]:.0f} min (~{stats_sommeil[7]/60:.1f} h)\n"
    )
    rapport_txt += "Écart-type                               : "
    rapport_txt += f"{stats_sommeil[8]:.0f} min\n"
    rapport_txt += (
        f"Durée moyenne en semaine   : "
        f"{stats_sommeil[9]:.0f} min (~{stats_sommeil[9]/60:.1f} h)\n"
    )
    rapport_txt += (
        f"Durée moyenne le week-end  : "
        f"{stats_sommeil[10]:.0f} min (~{stats_sommeil[10]/60:.1f} h)\n"
    )
    rapport_txt += f"Heure médiane coucher(1) : " f"{mins_to_hhmm(stats_sommeil[11])}\n"
    rapport_txt += (
        f"Heure médiane lever(1)     : " f"{mins_to_hhmm(stats_sommeil[12])}\n"
    )
    rapport_txt += "\n" + "-" * 110 + "\n"
    rapport_txt += "\n(1) Médiane moins sensible aux valeurs extrêmes"
    rapport_txt += "\n(2) Écart-type faible → sommeil régulier"

    logger.info("[REPORT] Rapport généré (%d caractères)", len(rapport_txt))
    return rapport_txt


def prepare_sleep_df(path: str | Path):
    """Charge et prépare un DataFrame de sommeil depuis un CSV."""
    try:
        logger.info("[PREPARE] Début pour %s", path)

        # --- 1. Lecture -------------------------------------------------
        if hasattr(path, "read"):  # cas Streamlit / fichier en mémoire
            path.seek(0)
            resultat = lire_fichier_pandas(path)
        else:
            resultat = lire_fichier_pandas(str(path))

        if resultat is None:
            logger.critical("[PREPARE] Abandon : échec de lecture %s", path)
            return pd.DataFrame()

        df = resultat["dataframe"]
        df.columns = df.columns.str.replace("\u00a0", " ", regex=False)
        df["qualite"] = df.apply(controle_validite, axis=1).fillna(pd.NA)

        logger.debug(
            "[PREPARE] Avant drop/rename : shape=%s, cols=%s",
            df.shape,
            df.columns.tolist(),
        )

        # --- 2. Normalisation colonnes ---------------------------------
        df = df.drop(columns=["Durée"], errors="ignore").rename(
            columns={
                "Sommeil 4 semaines": "date",
                "Heure de coucher": "coucher",
                "Heure de lever": "lever",
            }
        )
        logger.debug(
            "[PREPARE] Après drop/rename  : shape=%s, cols=%s",
            df.shape,
            df.columns.tolist(),
        )

        # --- 3. Nettoyage dates ----------------------------------------
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            logger.warning("[PREPARE] Certaines dates non parsées")
        if df["date"].isna().all():
            logger.warning("[PREPARE] Toutes les dates NaT pour %s", path)

        # --- 4. Nettoyage heures brutes --------------------------------
        df["raw_coucher"] = df["coucher"].astype(str).replace("--", pd.NA)
        df["raw_lever"] = df["lever"].astype(str).replace("--", pd.NA)

        for col in ["raw_coucher", "raw_lever"]:
            df[col] = (
                df[col]
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .str.strip("'")
                .str.replace(r"^(\d):", r"0\1:", regex=True)
            )

        tm_pattern = r"(\d{2}:\d{2}\s?[AP]M)"
        df["only_coucher"] = (
            df["raw_coucher"].str.extract(tm_pattern, expand=False).str.strip()
        )
        df["only_lever"] = (
            df["raw_lever"].str.extract(tm_pattern, expand=False).str.strip()
        )

        # --- 5. Conversion en Timestamp --------------------------------
        df["time_coucher"] = df["only_coucher"].apply(parse_time_ts)
        df["time_lever"] = df["only_lever"].apply(parse_time_ts)

        df["coucher"] = df.apply(combine, args=("time_coucher",), axis=1)
        df["lever"] = df.apply(combine, args=("time_lever",), axis=1)

        mask = df["lever"] <= df["coucher"]
        df.loc[mask, "lever"] += pd.Timedelta(days=1)

        # --- 6. Calculs dérivés ----------------------------------------
        df["duree"] = (df["lever"] - df["coucher"]).dt.total_seconds() / 60
        comps = (df["lever"] - df["coucher"]).dt.components
        df["heures"] = comps.days * 24 + comps.hours
        df["minutes"] = comps.minutes

        df["weekend"] = df["date"].dt.dayofweek >= 5

        df["coucher_dt"] = df["coucher"]
        df["lever_dt"] = df["lever"]
        df["mins_coucher"] = df["coucher_dt"].dt.hour * 60
        +df["coucher_dt"].dt.minute
        df["mins_lever"] = df["lever_dt"].dt.hour * 60
        +df["lever_dt"].dt.minute

        # --- 7. Validation colonnes ------------------------------------
        missing = set(COLS_REQ_STATS) - set(df.columns)
        if missing:
            logger.error("[PREPARE] Colonnes manquantes : %s", missing)
            raise KeyError(f"Colonnes manquantes : {missing}")

        logger.info(
            "[PREPARE] Terminé : %d lignes, %d colonnes",
            df.shape[0],
            df.shape[1],
        )
        return df
        print(df)
    except KeyError:
        raise
    except Exception:
        logger.exception("[PREPARE] Erreur inattendue pour %s", path)
        logger.critical(
            f"[PARSE_CSV] Échec : Impossible de lire {path}"
            + " avec les encodages et séparateurs fournis."
        )

    return pd.DataFrame()


def main():
    """Fonction principale exécutant l'analyse et le rapport de sommeil."""
    args = parse_args()
    logger.info("[MAIN] Arguments reçus : %s", args)

    input_path = Path(args.file)
    if not input_path.exists():
        logger.critical("[MAIN] Fichier introuvable : %s", input_path)
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    df = prepare_sleep_df(input_path)
    if df is None:
        logger.warning("[MAIN] Arrêt : prepare_sleep_df a renvoyé None")
        return

    stats_glob = calculer_stats_globales(df)
    stats_periode = calculer_stats_par_periode(df)
    heures_med = calculer_heures_medianes(df)
    stats_sommeil = stats_glob + stats_periode + heures_med

    rapport_txt = creer_rapport(
        stats_sommeil,
        df["date"].min().strftime("%A %d %B %Y"),
        df["date"].max().strftime("%A %d %B %Y"),
    )

    if args.viz:
        visualiser_distribution_duree(df)
        visualiser_evolution_duree(df)
        visualiser_coucher_vs_duree(df)

    if args.format in ["pdf", "both"]:
        ecrire_rapport_pdf(REPORT_DATA_FILE_PDF, rapport_txt)
    if args.format in ["txt", "both"]:
        ecrire_rapport_txt(REPORT_DATA_FILE_TXT, rapport_txt)

    logger.info("[MAIN] Analyse terminée avec succès")


if __name__ == "__main__":
    main()
