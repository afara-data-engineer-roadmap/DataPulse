# src/main.py

import pandas as pd
from pathlib import Path 
import logging

#configuration du logging (simple pour commencer)

# Définir le chemin vers dossier du projet

# Utilise Pathlib pour une meilleure gestion des chemins (multi-OS)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"  

SLEEP_DATA_FILE = DATA_DIR / "Sommeil.csv"

def load_sleep_data(file_path) : 
    """Charge les données de sommeil depuis un csv"""
    logging.info(f"Tentative de chargement des données depuis : {file_path}" )
    if not file_path.exists() : 
        logging.error(f"le fichier {file_path} n'esiste pas.")
        return None
    try : 
        df=pd.read_csv(file_path)
        logging.info(f"fichier csv chargé avec succés : {len(df)} lignes trouvées!")
        #afficher un aperçu pour vérifier
        logging.debug("apercçu des données chargées\n%s",df.head().to_string())
        return df 
    except Exception as e: 
        logging.error(f"Erreur lors du chargement du csv : {e} ")
        return None

def parse_duration(duration_str) : 
    """convertir une durée 'Xh Ymn' en minutes totales """
    if pd.isna(duration_str) or '.' not in duration_str : 
        return None 
    try : 
        hours = 0 
        minutes = 0 
        #sépare les heures et les minutes
        parts = duration_str.replace('min.','').strip().split("h ")
        if len (parts) == 2 :
            hours = int(parts[0]) 
            if parts[1]:
                minutes = int(parts(1)) 
            elif 'h' not in duration_str : 
                minutes = int(parts[0])
        total_minutes = hours * 60 + minutes
        
        return total_minutes
    except ValueError:
        logging.warning (f"impossible de parser la duréé : '{duration_str}'")
        return None
    

def process_sleep_data(df) : 
    """Traite les données de sommeil (ex: convertir duree en mn)
    """
    if df is None : 
        logging.warning("DataFrame vide , aucun traitement effectué")
        return None
    
    logging.info("Début du traitement de données...")
    processed_df = df.copy()
    
    #---- Tache principale pour jour 1 -----
    #convertir la colonne durée en minutes totales 
    processed_df['Duree_minutes'] = processed_df['Durée'].apply(parse_duration)
    logging.info("colonne 'Duree_minutes créée.")
    logging.debug("Apercu après ajout Dure_minutes" , processed_df.head().to_string() )
    
    # --- Autres traitements (pour plus tard) ---
   # TODO: Convertir 'Sommeil 4 semaines' en datetime
   # TODO: Convertir 'Heure de coucher' et 'Heure de lever' en datetime/time
   # TODO: Calculer la durée réelle entre coucher et lever
   # TODO: Nettoyer les noms de colonnes
   
    logging.info("Traitement des données terminé.")
    return processed_df

def generate_report (df) : 
    
    avg_duration = df['Duree_minutes'].mean()
    max_duration = df['Duree_minutes'].max()
    min_duration = df['Duree_minutes'].min()
    
    print ("\n--- Rapport Sommeil (Basique) ---")
    if not pd.isna(avg_duration) : 
        print(f"Durée moyenne du sommeil : {avg_duration:.0f} minutes (~{avg_duration/60:.1f} heures) ")
        print(f"Durée minimale : {min_duration:.0f} minutes")
        print(f"Durée maximale : {max_duration:.0f} minutes")
    else : 
        print(("Impossible de calculer la durée moyenne! (données manquantes ou erreur de parsing )"))
        print("---------------------------------\n")
        logging.info("Rapport généré et affiché.")

if __name__ == "__main__" : 
    logging.info ("Démarrage du script DataPulse")
    sleep_df = load_sleep_data(SLEEP_DATA_FILE )
    processed_sleep_df = process_sleep_data(sleep_df)
    generate_report(processed_sleep_df)
    logging.info ("Script DataPulse terminé")
    
