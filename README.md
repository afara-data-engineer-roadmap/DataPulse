# DataPulse - Analyseur de Données de Sommeil Personnelles

DataPulse est un script Python conçu pour analyser des données personnelles de sommeil exportées (par exemple, depuis une montre connectée ou une application) afin de fournir des statistiques et des visualisations sur les habitudes de sommeil.

## Objectifs du Projet

* Apprendre et mettre en pratique les concepts de Data Engineering et d'analyse de données avec Python et Pandas.
* Extraire des informations utiles à partir de mes propres données de sommeil.
* Développer un outil simple, modulaire et testé.

## Fonctionnalités Actuelles

* Lecture de données de sommeil depuis un fichier CSV (gère différents encodages/séparateurs).
* Nettoyage et préparation des données (parsing des dates et heures, gestion des valeurs manquantes).
* Calcul de la durée de sommeil, en gérant correctement les nuits à cheval sur minuit.
* Calcul de statistiques descriptives :
    * Durée moyenne, médiane, minimale, maximale, écart-type.
    * Nombre de nuits totales, normales (7-9h), courtes (<7h), longues (>9h).
    * Durée moyenne en semaine vs weekend.
    * Heure médiane de coucher et de lever.
* Génération d'un rapport textuel en console résumant ces statistiques.
* Visualisations graphiques (via Matplotlib/Seaborn) :
    * Distribution de la durée du sommeil (histogramme).
    * Évolution de la durée du sommeil au fil du temps (graphique linéaire).
    * Relation entre heure de coucher et durée du sommeil (nuage de points).
* Logging avancé des opérations (console et fichier rotatif).
* Tests unitaires (avec pytest) pour les fonctions utilitaires et de calcul.

## Structure du Projet

DataPulse/
├── data/                     # Données brutes (ex: Sommeil.csv, Sommeil_Test_Compatible.csv)
│   └── mon_script.log        # Fichier de log
├── src/                      # Code source Python
│   └── main.py               # Script principal
├── tests/                    # Tests unitaires
│   └── test_main.py
├── .gitignore                # Fichiers à ignorer par Git
├── pytest.ini                # Configuration pour pytest
└── README.md                 # Ce fichier


## Prérequis

* Python 3.x
* Librairies Python (à installer via `pip install -r requirements.txt` - *on créera ce fichier ensuite*) :
    * pandas
    * python-dateutil
    * matplotlib
    * seaborn
    * pytest (pour les tests)

## Utilisation

1.  Cloner le dépôt (si sur GitHub).
2.  S'assurer que les prérequis sont installés.
3.  Placer le fichier de données de sommeil (nommé `Sommeil.csv` ou `Sommeil_Test_Compatible.csv`) dans le dossier `data/`.
4.  Exécuter le script depuis la racine du projet :
    ```bash
    python src/main.py
    ```
5.  Les rapports textuels s'afficheront dans la console, les graphiques s'ouvriront dans des fenêtres séparées, et les logs détaillés seront dans `data/mon_script.log`.

## Prochaines Étapes Envisagées

Améliorer les visualisations (plus d'options, interactivité ?).
Ajouter une interface en ligne de commande (CLI) avec `argparse` pour plus de flexibilité (choix du fichier d'entrée, etc.).
Exporter le rapport dans un fichier (HTML, PDF ?).

---

*Ce projet est développé dans un but d'apprentissage personnel.*