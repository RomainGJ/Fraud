# Fraud Detection System

Un système de détection de fraude utilisant l'apprentissage automatique pour identifier les transactions frauduleuses.

## Fonctionnalités

- **Détection de fraude en temps réel** : Classification des transactions comme frauduleuses ou légitimes
- **Détection d'anomalies** : Identification de patterns anormaux dans les transactions
- **API REST** : Interface pour intégrer le système dans d'autres applications
- **Feature Engineering** : Création automatique de features à partir des données de transaction
- **Modèles multiples** : Support pour Random Forest et Régression Logistique

## Installation

### Prérequis
- Python 3.8+
- pip

### Configuration de l'environnement

```bash
# Cloner le repository
git clone <repository-url>
cd fraud_detection

# Créer et activer l'environnement virtuel
python3 -m venv fraud_detection_env
source fraud_detection_env/bin/activate  # Linux/Mac
# ou
fraud_detection_env\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Usage

### Entraînement du modèle

```bash
# Entraîner avec des données synthétiques
python main.py --train

# Entraîner avec vos propres données
python main.py --train --data path/to/your/data.csv

# Choisir le type de modèle
python main.py --train --model-type logistic_regression
```

### Prédiction

```bash
# Faire une prédiction sur des données d'exemple
python main.py --predict
```

### API Server

```bash
# Démarrer le serveur API
python main.py --api
```

L'API sera accessible sur `http://localhost:5000`

#### Endpoints API

- `GET /health` - Vérifier l'état de l'API
- `POST /predict` - Prédire une seule transaction
- `POST /predict/batch` - Prédire plusieurs transactions
- `GET /model/info` - Informations sur le modèle
- `GET /model/features` - Importance des features

#### Exemple de requête

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 850.0,
    "account_age_days": 15,
    "merchant_category": "online",
    "time_of_day": 23,
    "day_of_week": 6,
    "transaction_count_last_hour": 3,
    "average_transaction_amount": 120.0,
    "location_risk_score": 0.8,
    "payment_method": "credit"
  }'
```

## Structure du projet

```
fraud_detection/
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── fraud_detector.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py
│   └── utils/
├── tests/
├── data/
├── models/
├── notebooks/
├── config/
├── main.py
├── requirements.txt
└── README.md
```

## Features utilisées

Le système utilise plusieurs types de features pour la détection de fraude :

### Features temporelles
- Heure de la transaction
- Jour de la semaine
- Mois
- Indicateurs week-end/nuit

### Features de transaction
- Montant de la transaction
- Montant transformé (log, z-score)
- Catégorie de montant

### Features comportementales
- Statistiques utilisateur (moyenne, écart-type, etc.)
- Déviation par rapport au comportement habituel
- Ratio du montant actuel vs moyenne utilisateur

### Features de vitesse (velocity)
- Nombre de transactions dans les dernières heures
- Montant total dans les dernières heures

### Features de localisation et marchand
- Score de risque de localisation
- Score de risque marchand
- Volume de transactions marchand

## Tests

```bash
# Exécuter les tests
pytest tests/

# Avec couverture
pytest tests/ --cov=src
```

## Développement

### Code style

```bash
# Formater le code
black src/ tests/

# Vérifier le style
flake8 src/ tests/
```

## Performance

Le modèle est optimisé pour :
- **Précision** : Minimiser les faux positifs
- **Rappel** : Détecter le maximum de fraudes
- **Vitesse** : Prédictions en temps réel (<100ms)

## Métriques d'évaluation

- AUC-ROC
- Precision-Recall AUC
- Précision, Rappel, F1-Score
- Matrice de confusion
- Détection d'anomalies

## Licence

MIT License

## Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit les changements (`git commit -am 'Ajout d'une nouvelle feature'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Créer une Pull Request