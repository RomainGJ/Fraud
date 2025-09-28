# 🛡️ FraudGuard - Enterprise ML Fraud Detection Platform

FraudGuard est une plateforme complète de détection de fraude bancaire utilisant une stack open-source moderne avec MLflow, Airflow, Kubernetes, et un pipeline CI/CD GitLab.

## 🎯 Vue d'ensemble

FraudGuard combine l'apprentissage automatique avancé avec une infrastructure cloud-native pour détecter les fraudes en temps réel à l'échelle enterprise.

### 🔧 Stack Technique

| Composant | Technologie | Usage |
|-----------|------------|-------|
| **API** | FastAPI + Pydantic | API REST haute performance |
| **ML Tracking** | MLflow | Suivi d'expériences et registry de modèles |
| **Orchestration** | Apache Airflow | Pipelines ML automatisés |
| **Déploiement** | Kubernetes + Harbor | Scaling et gestion des containers |
| **CI/CD** | GitLab CI/CD | Intégration et déploiement continus |
| **Monitoring** | Prometheus + Grafana | Observabilité et alertes |
| **Stockage** | PostgreSQL + Redis | Base de données et cache |

## ✨ Fonctionnalités Enterprise

### 🚀 Détection de Fraude
- **Temps réel** : Prédictions < 100ms avec scaling automatique
- **Multi-modèles** : Random Forest + Isolation Forest pour détection d'anomalies
- **Feature Engineering** : +20 features automatiques (temporelles, comportementales, vélocité)
- **Validation avancée** : Pydantic pour validation des données d'entrée

### 🔄 MLOps Pipeline
- **Entraînement automatisé** : DAGs Airflow pour pipeline ML complet
- **Model Registry** : Versioning et déploiement via MLflow
- **A/B Testing** : Comparaison de modèles en production
- **Monitoring drift** : Détection automatique de dérive des modèles

### 🏗️ Infrastructure Cloud-Native
- **Auto-scaling** : HPA Kubernetes basé sur CPU/mémoire/trafic
- **Haute disponibilité** : Multi-réplicas avec load balancing
- **Zero-downtime** : Rolling deployments avec health checks
- **Observabilité** : Métriques business + infrastructure via Prometheus

### 🔒 Sécurité & Compliance
- **Network policies** : Isolation réseau Kubernetes
- **Security scanning** : Trivy + Bandit dans pipeline CI/CD
- **Secrets management** : Intégration avec vaults externes
- **Audit logs** : Traçabilité complète des prédictions

## 🚀 Déploiement

### Prérequis Infrastructure
- **Kubernetes** 1.20+ avec Helm 3
- **Docker** + Harbor Registry
- **GitLab** avec runners configurés
- **Domaines** configurés avec SSL

### 🐳 Déploiement Local (Développement)

```bash
# Cloner le repository
git clone https://github.com/RomainGJ/Fraud.git
cd fraud_detection

# Lancer la stack complète avec Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# Accès aux services
# API FraudGuard: http://localhost:8000
# MLflow: http://localhost:5000
# Airflow: http://localhost:8080 (admin/admin)
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### ☸️ Déploiement Production (Kubernetes)

```bash
# 1. Configuration du namespace
kubectl apply -f k8s/namespace.yaml

# 2. Déploiement de l'API
kubectl apply -f k8s/fraudguard-api-deployment.yaml

# 3. Configuration de l'auto-scaling
kubectl apply -f k8s/hpa.yaml

# 4. Exposition via Ingress
kubectl apply -f k8s/ingress.yaml

# 5. Monitoring
kubectl apply -f monitoring/
```

### 📊 Configuration MLflow + Airflow

```bash
# Variables d'environnement requises
export MLFLOW_TRACKING_URI=http://mlflow:5000
export AIRFLOW__CORE__EXECUTOR=CeleryExecutor
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow

# Initialisation Airflow
airflow db init
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@fraudguard.com --password admin
```

## 💻 Utilisation

### 🔧 Interface de Ligne de Commande

```bash
# Entraînement avec MLflow tracking
python main.py --train --model-type random_forest

# Test de l'API locale
python main.py --api

# Prédiction simple
python main.py --predict
```

### 🌐 API REST FastAPI

L'API FraudGuard expose plusieurs endpoints haute performance :

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Status de l'API et du modèle |
| `/api/v1/predict` | POST | Prédiction transaction unique |
| `/api/v1/predict/batch` | POST | Prédictions en lot |
| `/api/v1/model/info` | GET | Informations du modèle |
| `/api/v1/model/reload` | POST | Recharger le modèle |
| `/metrics` | GET | Métriques Prometheus |

#### 📝 Exemple d'Utilisation

```bash
# Prédiction unique
curl -X POST https://api.fraudguard.company.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "transaction_amount": 1250.0,
    "account_age_days": 15,
    "merchant_category": "online",
    "time_of_day": 23,
    "day_of_week": 6,
    "transaction_count_last_hour": 3,
    "average_transaction_amount": 120.0,
    "location_risk_score": 0.8,
    "payment_method": "credit",
    "transaction_id": "txn_12345"
  }'

# Réponse
{
  "transaction_id": "txn_12345",
  "is_fraud": true,
  "fraud_probability": 0.87,
  "is_anomaly": true,
  "risk_level": "HIGH",
  "confidence": 0.87,
  "prediction_time": "2024-01-15T10:30:00Z",
  "model_version": "v2.1.0"
}
```

### 🔄 Pipeline MLOps via Airflow

```bash
# Accéder à Airflow UI
open http://localhost:8080

# Déclencher manuellement le pipeline d'entraînement
airflow dags trigger fraud_detection_training_pipeline

# Voir les logs d'exécution
airflow logs fraud_detection_training_pipeline process_data 2024-01-15
```

### 📊 Monitoring et Observabilité

```bash
# Dashboard Grafana
open https://monitoring.fraudguard.company.com/grafana

# Métriques Prometheus
open https://monitoring.fraudguard.company.com/prometheus

# MLflow Experiments
open https://mlflow.fraudguard.company.com
```

## 📁 Architecture du Projet

```
fraudguard/
├── 🚀 src/                          # Code source principal
│   ├── api/                         # FastAPI application
│   │   ├── fastapi_app.py          # Application principale
│   │   └── __init__.py
│   ├── models/                      # Modèles ML
│   │   ├── fraud_detector.py       # Détecteur principal
│   │   └── __init__.py
│   ├── data_processing/             # Préprocessing des données
│   │   ├── preprocessor.py         # Pipeline de préparation
│   │   └── __init__.py
│   ├── features/                    # Feature engineering
│   │   ├── feature_engineering.py  # Création de features
│   │   └── __init__.py
│   └── mlflow_integration/          # Intégration MLflow
│       ├── experiment_tracker.py   # Tracking d'expériences
│       └── __init__.py
├── 🐳 docker/                       # Configuration Docker
│   ├── docker-compose.yml          # Stack complète de développement
│   └── Dockerfile.airflow          # Image Airflow personnalisée
├── ☸️ k8s/                          # Manifestes Kubernetes
│   ├── fraudguard-api-deployment.yaml  # Déploiement API
│   ├── hpa.yaml                    # Auto-scaling horizontal
│   ├── ingress.yaml                # Exposition externe
│   └── namespace.yaml              # Namespace isolé
├── 🔄 airflow/                      # Orchestration ML
│   ├── dags/                       # DAGs Airflow
│   │   ├── fraud_detection_pipeline.py     # Pipeline entraînement
│   │   └── fraud_detection_inference.py   # Pipeline inference
│   └── config/
│       └── airflow.cfg             # Configuration Airflow
├── 📊 monitoring/                   # Observabilité
│   ├── prometheus.yml              # Config monitoring
│   ├── fraud_detection_rules.yml   # Règles métier
│   ├── alerting_rules.yml          # Règles d'alerte
│   ├── alertmanager.yml            # Gestion des alertes
│   └── grafana/
│       └── fraudguard-dashboard.json   # Dashboard principal
├── 🔧 CI/CD
│   └── .gitlab-ci.yml              # Pipeline GitLab complet
├── 📋 Configuration
│   ├── requirements.txt            # Dépendances Python
│   ├── Dockerfile                  # Image de production
│   ├── .gitignore                 # Exclusions Git
│   └── README.md                  # Documentation
└── 🎯 Scripts
    ├── main.py                     # Point d'entrée principal
    └── tests/                      # Tests automatisés
        ├── unit/                   # Tests unitaires
        ├── integration/            # Tests d'intégration
        └── e2e/                    # Tests end-to-end
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