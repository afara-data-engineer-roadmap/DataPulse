# DataPulse – Analyseur de Données de Sommeil Personnelles

**DataPulse** est un outil Python open-source pour l’analyse et la visualisation de données de sommeil personnelles, conçu à la fois pour l’apprentissage du data engineering, de l’analyse de données, et pour obtenir des insights utiles sur ses propres habitudes de sommeil.

---

## 🚀 Objectifs du Projet

- Apprentissage pratique du data engineering, de la qualité logicielle (tests, CI) et de l’analyse de données avec Python et Pandas.
- Extraction d’indicateurs-clés à partir de données personnelles issues de montres connectées ou d’applications.
- Création d’un outil simple, modulaire, robuste, testé et automatisé.

---

## ⚡ Fonctionnalités Principales

- **Lecture intelligente** de fichiers CSV multi-encodages/séparateurs.
- **Nettoyage avancé** des données (dates/heures, valeurs manquantes, parsing robuste).
- **Calcul automatique** de la durée de sommeil (gestion des nuits à cheval sur minuit).
- **Statistiques détaillées :**
    - Durée moyenne, médiane, min, max, écart-type
    - Nombre de nuits totales, normales (7-9h), courtes (<7h), longues (>9h)
    - Comparaison semaine/weekend
    - Heures médianes de coucher/lever
- **Rapport textuel complet** (console ou exportable).
- **Visualisations graphiques :**
    - Histogrammes (distribution)
    - Séries temporelles (évolution)
    - Scatterplots (relation coucher/durée, etc.)
- **Logging détaillé** (console et fichier rotatif, niveaux paramétrables)
- **Tests unitaires** (Pytest) avec couverture, CI GitHub Actions
- **Qualité de code** : linting (`black`, `isort`, `flake8`, `ruff`)

---

## 📦 Structure du Projet

```
DataPulse/
├── data/                     # Données brutes (ex: Sommeil.csv, logs…)
├── src/                      # Code source Python
│   └── main.py
├── tests/                    # Tests unitaires et d’intégration
│   └── unit/
├── .github/
│   └── workflows/
│       └── ci.yml            # CI GitHub Actions (lint & tests)
├── requirements.txt          # Dépendances projet
├── pyproject.toml            # Config outils (Black, isort, flake8…)
├── pytest.ini                # Config Pytest
├── .flake8                   # (optionnel) Config flake8
└── README.md                 # Ce fichier'''
'''
# DataPulse – Analyseur de Données de Sommeil Personnelles

**DataPulse** est un outil Python open-source pour l’analyse et la visualisation de données de sommeil personnelles, conçu à la fois pour l’apprentissage du data engineering, de l’analyse de données, et pour obtenir des insights utiles sur ses propres habitudes de sommeil.

---

## 🚀 Objectifs du Projet

- Apprentissage pratique du data engineering, de la qualité logicielle (tests, CI) et de l’analyse de données avec Python et Pandas.
- Extraction d’indicateurs-clés à partir de données personnelles issues de montres connectées ou d’applications.
- Création d’un outil simple, modulaire, robuste, testé et automatisé.

---




