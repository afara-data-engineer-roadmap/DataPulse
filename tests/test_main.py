import pytest                   # Outil pour les tests
from main import mins_to_hhmm # On importe TA fonction
from main import parse_time_ts
from datetime import time 
import pandas as pd
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


        