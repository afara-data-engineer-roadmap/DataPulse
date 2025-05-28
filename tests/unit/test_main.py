from datetime import time
from io import StringIO

import pandas as pd
import pandas.testing as pd_testing
import pytest
from pandas import Timestamp

from main import (
    calculer_heures_medianes,
    calculer_stats_globales,
    calculer_stats_par_periode,
    mins_to_hhmm,
    parse_time_ts,
    prepare_sleep_df,
    extraire_heures,
    nettoyer_heures_brutes,
    nettoyer_dates,
    convertir_heure,
    corriger_lever_vs_coucher,
    valider_colonnes,
    ajouter_colonnes_derivees
)


def test_mins_to_hhmm_cas_simple():

    assert mins_to_hhmm(150) == "02:30"


def test_mins_to_hhmm_valeur_na():

    assert mins_to_hhmm(pd.NA) == "N/A"


def test_mins_to_hhmm_zero():

    assert mins_to_hhmm(0) == "00:00"


def test_mins_to_hhmm_edge():

    assert mins_to_hhmm(59) == "00:59"


def test_mins_to_hhmm_24h():

    assert mins_to_hhmm(1440) == "00:00"


def test_mins_to_hhmm_gt_24h():

    assert mins_to_hhmm(1500) == "01:00"


def test_mins_to_hhmm_60():

    assert mins_to_hhmm(60) == "01:00"


def test_parse_time_ts_valid_am():

    assert parse_time_ts("09:30") == time(9, 30)


def test_parse_time_ts_valid_pm():

    assert parse_time_ts("09:30 PM") == time(21, 30)


def test_parse_time_ts_midnight():

    assert parse_time_ts("12:00 AM") == time(0, 0)


def test_parse_time_ts_noon():

    assert parse_time_ts("12:00 PM") == time(12, 00)


def test_parse_time_ts_valeurs_invalides():
    assert parse_time_ts("not a time") is None
    assert parse_time_ts("") is None
    assert parse_time_ts(None) is None


def test_parse_time_na_input():

    assert parse_time_ts(pd.NA) is None


def test_calculer_stats_globales_simple():
    data = {"duree": [480.0, 420.0, 540.0, 600.0, 300.0, pd.NA]}
    df = pd.DataFrame(data)
    stats = calculer_stats_globales(df)
    assert stats[0] == 5  # total
    assert pytest.approx(stats[4], 0.1) == 468.0  # moyenne


def test_calculer_stats_globales_vide():
    df_vide = pd.DataFrame({"duree": []})
    stats = calculer_stats_globales(df_vide)
    assert stats == (0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_calculer_stats_globales_valeurs_extrêmes():
    df = pd.DataFrame({"duree": [0, 1440, 10000, -10, pd.NA]})
    stats = calculer_stats_globales(df)
    # Vérifie que la fonction calcule ss planter et donne des valeurs exactes
    assert stats[0] == 4  # nombre de valeurs valides (exclu pd.NA)
    assert stats[6] == -10  # minimale = -10 (même si absurde)
    assert stats[7] == 10000  # maximale


def test_calculer_stats_globales_aux_limites():
    # Durées  aux seuils 420 et 540 doivent être traitées comme "normales"
    df = pd.DataFrame({"duree": [420.0, 540.0]})
    (
        tot,
        normales,
        courtes,
        longues,
        moyenne,
        mediane,
        minimale,
        maximale,
        ecart_type,
    ) = calculer_stats_globales(df)
    assert tot == 2
    assert normales == 2
    assert courtes == 0
    assert longues == 0
    assert minimale == pytest.approx(420.0)
    assert maximale == pytest.approx(540.0)


def test_calculer_stats_globales_tout_court():
    # Ttes les durées sont strictement inférieures à 420 → tout dans "courtes"
    df = pd.DataFrame({"duree": [0.0, 1.0, 419.9]})
    tot, normales, courtes, longues, *_ = calculer_stats_globales(df)
    assert tot == 3
    assert normales == 0
    assert courtes == 3
    assert longues == 0


def test_calculer_stats_globales_tout_long():
    # Ttes les durées sont strictement supérieures à 540 → tout dans "longues"
    df = pd.DataFrame({"duree": [540.1, 1000.0]})
    tot, normales, courtes, longues, *_ = calculer_stats_globales(df)
    assert tot == 2
    assert normales == 0
    assert courtes == 0
    assert longues == 2


def test_calculer_stats_globales_uniquement_na():
    # Si toutes les valeurs sont manquantes, on doit obtenir le tuple de zéros
    df = pd.DataFrame({"duree": [pd.NA, pd.NA]})
    assert calculer_stats_globales(df) == (0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_calculer_stats_globales_moyenne_mediane_ecarts():
    # Vérif. que la moy., la méd. et l'éc.typ sont corrects sur un petit jeu
    df = pd.DataFrame({"duree": [400.0, 450.0, 500.0]})
    (
        tot,
        normales,
        courtes,
        longues,
        moyenne,
        mediane,
        minimale,
        maximale,
        ecart_type,
    ) = calculer_stats_globales(df)
    # tot = 3, courtes = 1 (<420), normales = 2 (>=420 et <=540), longues = 0
    assert tot == 3
    assert courtes == 1
    assert normales == 2
    assert longues == 0
    # moyenne = (400+450+500)/3 = 450
    assert moyenne == pytest.approx(450.0)
    # médiane = 450
    assert mediane == pytest.approx(450.0)
    # minimale et maximale
    assert minimale == pytest.approx(400.0)
    assert maximale == pytest.approx(500.0)
    # écart-type
    expected_std = df["duree"].std()
    assert ecart_type == pytest.approx(expected_std)


def test_calculer_stats_par_periode_simple():
    data = {
        "date": [
            Timestamp("2025-05-01"),
            Timestamp("2025-05-02"),
            Timestamp("2025-05-03"),
            Timestamp("2025-05-04"),
            Timestamp("2025-05-05"),
        ],
        "duree": [450.0, 470.0, 550.0, 570.0, 460.0],
    }
    df = pd.DataFrame(data)
    semaine, weekend = calculer_stats_par_periode(df)
    assert pytest.approx(semaine) == 460
    assert pytest.approx(weekend) == 560


def test_calculer_heures_medianes():
    data = {"mins_coucher": [1320, 1410, 1380], "mins_lever": [420, 480, 450]}
    df = pd.DataFrame(data)
    median_coucher, median_lever = calculer_heures_medianes(df)
    assert median_coucher == 1380
    assert median_lever == 450


def test_prepare_sleep_df_simple():
    csv_data = StringIO(
        """Sommeil 4 semaines;Heure de coucher;Heure de lever;Durée
        2025-05-01;10:00 PM;06:00 AM;480
        2025-05-02;11:00 PM;07:00 AM;480
        """
    )
    df = prepare_sleep_df(csv_data)

    # Vérifie que le DataFrame n'est pas None
    assert df is not None

    # Vérifie que les colonnes attendues existent
    expected_cols = [
        "date",
        "coucher",
        "lever",
        "duree",
        "heures",
        "minutes",
        "qualite",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Vérifie que les dates sont bien converties en datetime
    assert pd.api.types.is_datetime64_any_dtype(df["date"])

    # Vérifie que les durées sont numériques et cohérentes
    assert df["duree"].dtype.kind in "fi"  # float ou int

    # Vérifie la première valeur de 'qualite' est 'valide'
    assert df["qualite"].iloc[0] == "valide"


def test_prepare_sleep_df_valeurs_manquantes():
    # Simuler un DataFrame avec des valeurs '--' ou NaN dans les colonnes heure
    from io import StringIO

    csv_data = StringIO(
        """Sommeil 4 semaines;Heure de coucher;Heure de lever;Durée
2025-05-01;-- ;08:00 AM;480
2025-05-02;10:00 PM;--;450
2025-05-03;10:00 PM;06:00 AM;--"""
    )

    df_result = prepare_sleep_df(csv_data)

    # Vérifier que la colonne 'qualite' détecte les lignes invalides
    assert df_result["qualite"].str.contains("manquante").any()


def test_prepare_sleep_df_fichier_vide():
    csv_vide = StringIO("")
    df = prepare_sleep_df(csv_vide)
    # On s'attend à un DataFrame vide, pas None, avec colonnes attendues
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_prepare_sleep_df_format_heure_incoherent():
    csv_data = StringIO(
        """Sommeil 4 semaines;Heure de coucher;Heure de lever;Durée
        2025-05-01;25:00 PM;06:00 AM;480
        2025-05-02;10:00 PM;99:99 AM;480
        2025-05-03;10 PM;06:00 AM;480
        2025-05-04;-- ;07:00 AM;450
        2025-05-05;22:00 PM;-- ;460
        2025-05-06;abc;def;470
        """
    )
    df = prepare_sleep_df(csv_data)

    # 1) Les colonnes existent
    for col in ("date", "coucher", "lever", "duree", "qualite"):
        assert col in df.columns

    # 2) Pr chq lg dont le parsing a échoué, 'coucher' ou 'lever' doit être NaT
    failed_coucher = df["coucher"].isna()
    failed_lever = df["lever"].isna()
    # on s’attend à au moins un NaT dans chaque cas
    assert failed_coucher.any() or failed_lever.any()

    # 3) Les lgs correct. format. (ex. idx 2 = "10 PM;06:00 AM") donnent Tstamp
    # ici la 3ᵉ lg (index 2 dans l’ordre du DataFrame après parsing) est valide
    # on vérifie qu’elle n’est pas NaT
    valid_idx = df[(df["only_coucher"].notna()) & (df["only_lever"].notna())].index
    assert any(pd.notna(df.loc[i, "coucher"]) for i in valid_idx)
    assert any(pd.notna(df.loc[i, "lever"]) for i in valid_idx)


def test_prepare_sleep_df_format_heure_melange():
    csv_data = StringIO(
        """Sommeil 4 semaines;Heure de coucher;Heure de lever;Durée
        2025-05-01;10:00 PM;06:00 AM;480
        2025-05-02;22:00;06:00;460
        2025-05-03;22h00;06h00;470
        2025-05-04;10 PM;6 AM;450
        2025-05-05;-- ;-- ;0
        """
    )
    df = prepare_sleep_df(csv_data)

    # 1) La première ligne était parfaite : on doit avoir un timestamp
    assert isinstance(df.loc[0, "coucher"], Timestamp)
    assert isinstance(df.loc[0, "lever"], Timestamp)

    # 2) La dern lg a des "--" : parse_time_ts donne None → combine() donne NaT
    assert pd.isna(df.loc[4, "coucher"])
    assert pd.isna(df.loc[4, "lever"])

    # 3) Pour les lignes intermédiaires, on s’attend que dateutil.parse gère :
    #    - "22:00" et "06:00" → Timestamp
    #    - "22h00"/"06h00" → svt reconnu par dateutil, tt dépend de la version
    #    - "10 PM"/"6 AM" → Timestamp
    for idx in (1, 2, 3):
        # si parsing échoue, la colonne sera NaT ; sinon ce sera un Timestamp
        val_c = df.loc[idx, "coucher"]
        val_l = df.loc[idx, "lever"]
        assert (pd.isna(val_c) and pd.isna(val_l)) or (
            isinstance(val_c, Timestamp) and isinstance(val_l, Timestamp)
        )


def test_demonstration_assert_frame_equal():
    print("Démontre le fonctionnement de pandas.testing.asser_frame_equal")
    print("Démo assert_frame_equal")

    df_obtenu1 = pd.DataFrame({"colA": [1, 2], "colB": ["x", "y"]})
    df_attendu1 = pd.DataFrame({"colA": [1, 2], "colB": ["x", "y"]})
    print("\nTest avec dataframes identiques :")
    try:
        pd_testing.assert_frame_equal(df_obtenu1, df_attendu1)
        print("Succés ils sont identiques")
    except AssertionError as e:
        print(f"-> Echec inattendu : {e}")

    df_obtenu2 = pd.DataFrame({"colA": [1, 99], "colB": ["x", "y"]})
    df_attendu2 = pd.DataFrame({"colA": [1, 2], "colB": ["x", "y"]})
    print("\nTest avec valeurs différentes :")
    try:
        pd_testing.assert_frame_equal(df_obtenu2, df_attendu2)
        print("-> Succés inattendu !")
    except AssertionError as e:
        print(f"-> Echec attendu message d'erreur : {e}")
        print(e)


def test_calcul_duree_composantes():
    """
    Teste le calcul durée et ses composantes (heures, mins, weekend, etc.)
    à partir de Timestamps de coucher et lever déjà préparés.
    Simule la fin de la logique de prepare_sleep_df.
    """
    # 1. ARRANGE: Crée un DataFrame d'entrée simple AVEC des Timestamps
    data_entree = {
        "date": [Timestamp("2025-05-06"), Timestamp("2025-05-07")],
        "coucher": [
            Timestamp("2025-05-06 01:00:00"),
            Timestamp("2025-05-07 23:00:00"),
        ],
        # Lever déjà ajusté (pour le 2ème cas, on est le 8 mai)
        "lever": [
            Timestamp("2025-05-06 08:30:00"),
            Timestamp("2025-05-08 06:45:00"),
        ],
    }
    df_entree = pd.DataFrame(data_entree)

    # 2. ACT: App la logique de calc (cop/adapt de ta fction prepare_sleep_df)
    df_obtenu = df_entree.copy()
    delta_duree = df_obtenu["lever"] - df_obtenu["coucher"]
    df_obtenu["duree"] = delta_duree.dt.total_seconds() / 60
    comps = delta_duree.dt.components
    df_obtenu["heures"] = comps.days * 24 + comps.hours
    df_obtenu["minutes"] = comps.minutes
    df_obtenu["weekend"] = df_obtenu["date"].dt.dayofweek >= 5  # Lun=0, Dim=6
    df_obtenu["coucher_dt"] = df_obtenu["coucher"]
    df_obtenu["lever_dt"] = df_obtenu["lever"]
    df_obtenu["mins_coucher"] = (
        df_obtenu["coucher_dt"].dt.hour * 60 + df_obtenu["coucher_dt"].dt.minute
    )
    df_obtenu["mins_lever"] = (
        df_obtenu["lever_dt"].dt.hour * 60 + df_obtenu["lever_dt"].dt.minute
    )

    # 3. ARRANGE: Définis le DataFrame EXACT que tu attends en sortie
    data_attendu = {
        "date": [Timestamp("2025-05-06"), Timestamp("2025-05-07")],
        "coucher": [
            Timestamp("2025-05-06 01:00:00"),
            Timestamp("2025-05-07 23:00:00"),
        ],
        "lever": [
            Timestamp("2025-05-06 08:30:00"),
            Timestamp("2025-05-08 06:45:00"),
        ],
        "duree": [450.0, 465.0],  # 7h30m = 450 min | 7h45m = 465 min
        "heures": [7, 7],
        "minutes": [30, 45],
        "weekend": [False, False],  # Mardi (1) et Mercredi (2) => pas weekend
        "coucher_dt": [
            Timestamp("2025-05-06 01:00:00"),
            Timestamp("2025-05-07 23:00:00"),
        ],
        "lever_dt": [
            Timestamp("2025-05-06 08:30:00"),
            Timestamp("2025-05-08 06:45:00"),
        ],
        "mins_coucher": [60, 1380],  # 1*60 | 23*60
        "mins_lever": [510, 405],  # 8*60+30 | 6*60+45
    }
    df_attendu = pd.DataFrame(data_attendu)
    # Assurer le même ordre de colonnes pour la comparaison (important !)
    df_attendu = df_attendu[df_obtenu.columns]

    # 4. ASSERT: Compare les deux DataFrames
    pd_testing.assert_frame_equal(df_obtenu, df_attendu, check_dtype=False)


def test_nettoyer_dates_basique():
    """
    Teste la fonction nettoyer_dates sur des dates valides et invalides
    rencontrées dans tes fichiers CSV réels.
    """
    serie = pd.Series(['2025-05-26', 'not a date', None, '--'])
    result = nettoyer_dates(serie)
    assert pd.notna(result[0])    # Date ISO
    assert pd.isna(result[1])     # Invalide
    assert pd.isna(result[2])     # None
    assert pd.isna(result[3])     # --


def test_nettoyer_heures_brutes():
    """
    Teste la fonction nettoyer_heures_brutes sur des heures avec espaces, '--', apostrophes.
    - Doit normaliser les heures valides.
    - Doit convertir '--' en NA.
    """
    serie = pd.Series([' 22:10 ', '--', "7:05", "'23:01'", " 10:20 PM "])
    result = nettoyer_heures_brutes(serie)
    assert result[0] == "22:10"     # Retrait espaces
    assert pd.isna(result[1])       # '--' => NA
    assert result[2] == "07:05"     # Ajout zéro devant
    assert result[3] == "23:01"     # Retrait apostrophes
    assert result[4] == "10:20 PM"  # Espace et majuscule conservés


def test_extraire_heures_nettoyees():
    """
    Teste extraire_heures sur les valeurs d'heures déjà nettoyées.
    Après nettoyage, seuls 'HH:MM AM/PM' ou NA sont possibles.
    """
    serie = pd.Series([
        "10:29 PM",    # Format AM/PM valide
        "08:28 AM",    # Format AM/PM valide
        pd.NA,         # Valeur manquante après nettoyage
        "12:22 AM",    # Format AM/PM valide
        "11:43 PM",    # Format AM/PM valide
    ])
    result = extraire_heures(serie)
    assert result[0] == "10:29 PM"
    assert result[1] == "08:28 AM"
    assert pd.isna(result[2])
    assert result[3] == "12:22 AM"
    assert result[4] == "11:43 PM"

def test_convertir_heure_am_pm():
    """
    Vérifie que convertir_heure transforme des chaînes AM/PM en objets time.
    """
    serie = pd.Series(["10:29 PM", "8:28 AM", "12:00 PM", "12:00 AM"])
    result = convertir_heure(serie)
    assert result[0] == time(22, 29)
    assert result[1] == time(8, 28)
    assert result[2] == time(12, 0)
    assert result[3] == time(0, 0)

def test_convertir_heure_na_et_invalide():
    """
    Vérifie que convertir_heure gère les valeurs manquantes ou invalides.
    """
    serie = pd.Series([pd.NA, "--", None, "", "bidule"])
    result = convertir_heure(serie)
    assert pd.isna(result[0])
    assert pd.isna(result[1])
    assert pd.isna(result[2])
    assert pd.isna(result[3])
    assert pd.isna(result[4])

def test_convertir_heure_melange_valeurs():
    """
    Teste un mélange de valeurs valides et invalides.
    """
    serie = pd.Series(["11:45 PM", "bidule", "1:00 AM", pd.NA])
    result = convertir_heure(serie)
    assert result[0] == time(23, 45)
    assert pd.isna(result[1])
    assert result[2] == time(1, 0)
    assert pd.isna(result[3])
    
def test_corriger_lever_vs_coucher_ajustement():
    """
    Vérifie que les lignes où 'lever' <= 'coucher' sont bien corrigées (+1 jour sur 'lever').
    """
    df = pd.DataFrame({
        "coucher": [pd.Timestamp("2024-05-28 23:00:00"), pd.Timestamp("2024-05-29 23:30:00")],
        "lever":   [pd.Timestamp("2024-05-28 06:30:00"), pd.Timestamp("2024-05-30 07:00:00")],
    })
    df_corrige = corriger_lever_vs_coucher(df.copy())
    # 1ʳᵉ ligne doit être corrigée (lever < coucher)
    assert df_corrige.loc[0, "lever"] == pd.Timestamp("2024-05-29 06:30:00")
    # 2ᵉ ligne ne change pas (lever > coucher)
    assert df_corrige.loc[1, "lever"] == pd.Timestamp("2024-05-30 07:00:00")

def test_corriger_lever_vs_coucher_aucune_correction():
    """
    Vérifie que s'il n'y a rien à corriger, la DataFrame reste inchangée.
    """
    df = pd.DataFrame({
        "coucher": [pd.Timestamp("2024-05-28 22:00:00"), pd.Timestamp("2024-05-28 23:00:00")],
        "lever":   [pd.Timestamp("2024-05-29 06:00:00"), pd.Timestamp("2024-05-29 07:30:00")],
    })
    df_corrige = corriger_lever_vs_coucher(df.copy())
    pd.testing.assert_frame_equal(df, df_corrige)

def test_corriger_lever_vs_coucher_egalite():
    """
    Vérifie que si 'lever' == 'coucher', la correction s'applique aussi.
    """
    df = pd.DataFrame({
        "coucher": [pd.Timestamp("2024-05-28 22:00:00")],
        "lever":   [pd.Timestamp("2024-05-28 22:00:00")],
    })
    df_corrige = corriger_lever_vs_coucher(df.copy())
    assert df_corrige.loc[0, "lever"] == pd.Timestamp("2024-05-29 22:00:00")

def test_ajouter_colonnes_derivees_valeurs_attendues():
    """
    Vérifie que les colonnes dérivées (duree, heures, minutes, weekend, etc.)
    sont correctement ajoutées et calculées.
    """
    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-05-28"), pd.Timestamp("2024-05-25")],  # Mardi, Samedi
        "coucher": [pd.Timestamp("2024-05-28 22:00:00"), pd.Timestamp("2024-05-25 23:30:00")],
        "lever": [pd.Timestamp("2024-05-29 06:30:00"), pd.Timestamp("2024-05-26 07:00:00")],
    })
    df_derive = ajouter_colonnes_derivees(df.copy())
    # Durée en minutes
    assert df_derive.loc[0, "duree"] == 510  # 8h30 = 510 min
    assert df_derive.loc[1, "duree"] == 450  # 7h30 = 450 min
    # Heures et minutes
    assert df_derive.loc[0, "heures"] == 8
    assert df_derive.loc[0, "minutes"] == 30
    assert df_derive.loc[1, "heures"] == 7
    assert df_derive.loc[1, "minutes"] == 30
    # Weekend flag (Samedi = True)
    assert not df_derive.loc[0, "weekend"]  # Mardi
    assert df_derive.loc[1, "weekend"]      # Samedi
    # Vérifie coucher_dt et lever_dt
    assert df_derive.loc[0, "coucher_dt"] == pd.Timestamp("2024-05-28 22:00:00")
    assert df_derive.loc[0, "lever_dt"] == pd.Timestamp("2024-05-29 06:30:00")
    # Minutes depuis minuit
    assert df_derive.loc[0, "mins_coucher"] == 22 * 60
    assert df_derive.loc[0, "mins_lever"] == 6 * 60 + 30

def test_valider_colonnes_ok():
    """
    Vérifie que valider_colonnes ne lève pas d'erreur si tout est présent.
    """
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    try:
        valider_colonnes(df, ["a", "b"])
    except KeyError:
        assert False, "KeyError ne doit pas être levé si tout est ok"

def test_valider_colonnes_erreur():
    """
    Vérifie que valider_colonnes lève une erreur si une colonne manque.
    """
    df = pd.DataFrame({"a": [1], "b": [2]})
    try:
        valider_colonnes(df, ["a", "b", "c"])
    except KeyError as e:
        assert "Colonnes manquantes" in str(e)
    else:
        assert False, "KeyError doit être levé si une colonne manque"

def test_lire_et_normaliser_csv_valide_et_invalide(tmp_path):
    contenu = (
        "Sommeil 4 semaines,Durée,Heure de coucher,Heure de lever\n"
        "2025-05-26,9h 43min.,10:29 PM,8:28 AM\n"
        "2025-05-12,--,--,--\n"
    )
    fichier = tmp_path / "sommeil.csv"
    fichier.write_text(contenu, encoding="utf-8")

    # Mock lire_fichier_pandas pour retourner DataFrame attendu
    import main
    def mock_lire_fichier_pandas(path):
        return {"dataframe": pd.read_csv(path)}
    original = main.lire_fichier_pandas
    main.lire_fichier_pandas = mock_lire_fichier_pandas

    try:
        df = main.lire_et_normaliser_csv(fichier)
    finally:
        main.lire_fichier_pandas = original

    assert df.loc[0, "qualite"] == "valide"
    assert df.loc[1, "qualite"] == "Durée manquante; Heure de coucher manquante; Heure de lever manquante"

