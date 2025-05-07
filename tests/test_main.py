import pandas as pd
from pandas import Timestamp
import pandas.testing as pd_testing
import pytest                   # Outil pour les tests
from main import mins_to_hhmm # On importe TA fonction
from main import parse_time_ts
from main import calculer_stats_globales # On importe TA fonction
from main import calculer_stats_par_periode # On importe TA fonction

from datetime import time 

def test_mins_to_hhmm_cas_simple(): 
    minutes_entree = 150 
    resultat_attendu = "02:30"
    resultat_obtenu = mins_to_hhmm(minutes_entree)
    assert resultat_obtenu == resultat_attendu

def test_mins_to_hhmm_valeur_na(): 
    valeur = pd.NA 
    resultat_attendu = "N/A"
    resultat_obtenu = mins_to_hhmm(valeur)
    assert resultat_obtenu == resultat_attendu


def test_mins_to_hhmm_zero():
    assert mins_to_hhmm(0) == "00:00"
    
def test_mins_to_hhmm_edge():
    assert mins_to_hhmm(59) == "00:59"
    
def test_mins_to_hhmm_24h():
    assert mins_to_hhmm (1440) == "00:00"

def test_mins_to_hhmm_gt_24h():
    assert mins_to_hhmm (1500) == "01:00"
    
def test_mins_to_hhmm_60():
    assert mins_to_hhmm(60) == "01:00"
    

def test_parse_time_ts_valid_am():
    assert parse_time_ts("09:30") == time (9,30)
    
    
def test_parse_time_ts_valid_pm():
    assert parse_time_ts("09:30 PM") == time (21,30)    
    
def test_parse_time_ts_midnight():
    assert parse_time_ts("12:00 AM") == time (0,0)

def test_parse_time_ts_noon():
    assert parse_time_ts("12:00 PM") == time (12,00)
    
    
def test_parse_time_invalid_string():
    assert parse_time_ts("Hello world!") is None
    
def test_parse_time_na_input():
    assert parse_time_ts(pd.NA) is None
    
def test_calculer_stats_globales_cas_simple():
    #Teste le calcul des statisq=tiuqes globales sur un cas simple 
    
    data_entree = {"duree" : [480.0,420.0,540.0,600.0,300.0,pd.NA]}
    df_test_entree = pd.DataFrame(data_entree)
    
    stats_attendues = (
        5,
        3,
        1,
        1,
        468.0,
        480.0,
        300.0,
        600.0,
        df_test_entree['duree'].std()
        )
    
    stats_obtenues = calculer_stats_globales(df_test_entree)
    assert stats_obtenues[0:4] == stats_attendues[0:4] 
    assert stats_obtenues[4] == pytest.approx(stats_attendues[4]) # moyenne
    assert stats_obtenues[5] == pytest.approx(stats_attendues[5]) # mediane (peut être .5)
    assert stats_obtenues[6] == pytest.approx(stats_attendues[6]) # minimale
    assert stats_obtenues[7] == pytest.approx(stats_attendues[7]) # maximale
    assert stats_obtenues[8] == pytest.approx(stats_attendues[8]) # ecart_type
    
def test_calculer_stats_par_periode_simple():
    data_entree = {"date" : [Timestamp('2025-05-01'),
                             Timestamp('2025-05-02'),
                             Timestamp('2025-05-03'),
                             Timestamp('2025-05-04'),
                             Timestamp('2025-05-05')                             
                             
                             ],
                    "duree" : [450.0,470.0,550.0,570.0,460.0]}
        
    df_test = pd.DataFrame(data_entree)
    moyenne_semaine_attendue = 460.0
    moyenne_weekend_attendue = 560.0
    
    moyenne_semaine_obtenue, moyenne_weekend_obtenue  = calculer_stats_par_periode(df_test)
    
    assert moyenne_semaine_obtenue == pytest.approx(moyenne_semaine_attendue)
    assert moyenne_weekend_obtenue == pytest.approx(moyenne_weekend_attendue)
    
def test_demonstration_assert_frame_equal() : 
    print("Démontre le fonctionnement de pandas.testing.asser_frame_equal")
    print("Démo assert_frame_equal")
    
    df_obtenu1 = pd.DataFrame({'colA' : [1,2], 'colB' : ['x','y']})
    df_attendu1 = pd.DataFrame({'colA' : [1,2], 'colB' : ['x','y']})
    print("\nTest avec dataframes identiques :")
    try :
        pd_testing.assert_frame_equal(df_obtenu1,df_attendu1)
        print("Succés ils sont identiques")
    except AssertionError as e : 
        print (f"-> Echec inattendu : {e}")
    
    df_obtenu2 = pd.DataFrame({'colA' : [1,99], 'colB' : ['x','y']})
    df_attendu2 = pd.DataFrame({'colA' : [1,2], 'colB' : ['x','y']})
    print("\nTest avec valeurs différentes :")
    try :
        pd_testing.assert_frame_equal(df_obtenu2,df_attendu2)
        print("-> Succés inattendu !")
    except AssertionError as e : 
        print (f"-> Echec attendu message d'erreur : {e}")
        print(e)
   
        
    df_obtenu3 = pd.DataFrame({'colA' : [1,2], 'colB' : ['10','20']}) 
    df_attendu3 = pd.DataFrame({'colA' : [1,2], 'colB' : [10,20]})
    print("\nTest avec valeurs différentes :")
    try :
        pd_testing.assert_frame_equal(df_obtenu3,df_attendu3)
        print("-> Succés inattendu !")
    except AssertionError as e : 
       print (f"-> Echec attendu message d'erreur : {e}")
       print(e)

def test_calcul_duree_composantes():
    """
    Teste le calcul de la durée et de ses composantes (heures, mins, weekend, etc.)
    à partir de Timestamps de coucher et lever déjà préparés.
    Simule la fin de la logique de prepare_sleep_df.
    """
    # 1. ARRANGE: Crée un DataFrame d'entrée simple AVEC des Timestamps
    data_entree = {
        'date': [Timestamp('2025-05-06'), Timestamp('2025-05-07')], # Mardi, Mercredi
        'coucher': [Timestamp('2025-05-06 01:00:00'), Timestamp('2025-05-07 23:00:00')],
        # Lever déjà ajusté (pour le 2ème cas, on est le 8 mai)
        'lever': [Timestamp('2025-05-06 08:30:00'), Timestamp('2025-05-08 06:45:00')]
    }
    df_entree = pd.DataFrame(data_entree)

    # 2. ACT: Applique la logique de calcul (copiée/adaptée de ta fonction prepare_sleep_df)
    df_obtenu = df_entree.copy()
    delta_duree = (df_obtenu['lever'] - df_obtenu['coucher'])
    df_obtenu['duree'] = delta_duree.dt.total_seconds() / 60
    comps = delta_duree.dt.components
    df_obtenu['heures']  = comps.days * 24 + comps.hours
    df_obtenu['minutes'] = comps.minutes
    df_obtenu['weekend'] = df_obtenu['date'].dt.dayofweek >= 5 # Lundi=0, Dimanche=6
    df_obtenu['coucher_dt'] = df_obtenu['coucher']
    df_obtenu['lever_dt'] = df_obtenu['lever']
    df_obtenu['mins_coucher'] = df_obtenu['coucher_dt'].dt.hour * 60 + df_obtenu['coucher_dt'].dt.minute
    df_obtenu['mins_lever'] = df_obtenu['lever_dt'].dt.hour * 60 + df_obtenu['lever_dt'].dt.minute

    # 3. ARRANGE: Définis le DataFrame EXACT que tu attends en sortie
    data_attendu = {
        'date': [Timestamp('2025-05-06'), Timestamp('2025-05-07')],
        'coucher': [Timestamp('2025-05-06 01:00:00'), Timestamp('2025-05-07 23:00:00')],
        'lever': [Timestamp('2025-05-06 08:30:00'), Timestamp('2025-05-08 06:45:00')],
        'duree': [450.0, 465.0],      # 7h30m = 450 min | 7h45m = 465 min
        'heures': [7, 7],
        'minutes': [30, 45],
        'weekend': [False, False],    # Mardi (1) et Mercredi (2) ne sont pas weekend
        'coucher_dt': [Timestamp('2025-05-06 01:00:00'), Timestamp('2025-05-07 23:00:00')],
        'lever_dt': [Timestamp('2025-05-06 08:30:00'), Timestamp('2025-05-08 06:45:00')],
        'mins_coucher': [60, 1380],   # 1*60 | 23*60
        'mins_lever': [510, 405]      # 8*60+30 | 6*60+45
    }
    df_attendu = pd.DataFrame(data_attendu)
    # Assurer le même ordre de colonnes pour la comparaison (important !)
    df_attendu = df_attendu[df_obtenu.columns]

    # 4. ASSERT: Compare les deux DataFrames
    pd_testing.assert_frame_equal(df_obtenu, df_attendu)

        