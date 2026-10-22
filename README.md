# Détection de la fraude bancaire

Application web de scoring du risque de fraude sur des transactions bancaires. Elle compare plusieurs modèles de machine learning, sélectionne automatiquement le meilleur et classe chaque transaction en trois niveaux : **Normal**, **Suspect** ou **Fraude**.

Développée avec Python, scikit-learn et Streamlit.

## Fonctionnalités

| Page | Ce qu'elle fait |
| --- | --- |
| Résumé exécutif | Indicateurs clés, répartition des prédictions, classement des modèles |
| Données | Aperçu du jeu de données, structure des colonnes, valeurs manquantes, statistiques |
| Analyse des variables | Distributions par niveau de risque, matrice de corrélation |
| Modèles IA | Comparaison des métriques, matrice de confusion, courbes ROC et Precision-Recall |
| Prédiction par entrée | Score de risque et décision pour chaque transaction |
| Simulation transaction | Saisie manuelle d'une transaction et calcul immédiat de son score |
| Importance des variables | Variables qui pèsent le plus dans la décision |
| Export | Téléchargement des résultats au format CSV |

## Démarche

1. **Chargement des données** : import d'un fichier CSV, ou jeu de démonstration généré automatiquement si aucun fichier n'est fourni.
2. **Préparation** :
   - normalisation des noms de colonnes ;
   - détection automatique de la colonne cible (`isFraud`, `fraud`, `fraude`, `status`, `label`…) et conversion en variable binaire ;
   - exclusion des identifiants et des colonnes texte à trop forte cardinalité.
3. **Prétraitement** : imputation des valeurs manquantes, standardisation des variables numériques, encodage one-hot des variables catégorielles, le tout dans un pipeline scikit-learn.
4. **Modélisation** : trois modèles entraînés et comparés sur un découpage stratifié 75 % / 25 % :
   - Régression logistique ;
   - Random Forest ;
   - Extra Trees.

   Les classes sont pondérées (`class_weight="balanced"`), car les fraudes sont rares.
5. **Sélection** : le meilleur modèle est choisi selon le **PR-AUC** (aire sous la courbe Precision-Recall), plus adapté que l'accuracy quand la classe à détecter est minoritaire.
6. **Décision** : le score de probabilité est converti en trois niveaux de risque, avec des seuils réglables dans l'interface (par défaut, Suspect à partir de 30 % et Fraude à partir de 70 %).

## Format des données attendu

L'application accepte tout fichier CSV qui contient une colonne cible identifiable. Elle est pensée pour des transactions au format suivant :

| Colonne | Description |
| --- | --- |
| `step` | Unité de temps de la transaction |
| `type` | Type d'opération (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN) |
| `amount` | Montant |
| `oldbalanceOrg`, `newbalanceOrig` | Solde du compte émetteur avant et après |
| `oldbalanceDest`, `newbalanceDest` | Solde du compte destinataire avant et après |
| `isFraud` | Cible : 1 = fraude, 0 = normal |

Sans fichier importé, l'application génère **3 000 transactions synthétiques** où la fraude est plus probable pour les virements et retraits, les montants élevés et les comptes vidés. Les résultats obtenus sur ce jeu de démonstration servent à illustrer le fonctionnement, pas à mesurer une performance réelle.

## Installation et lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre ensuite dans le navigateur à l'adresse http://localhost:8501.

## Technologies

Python · pandas · NumPy · scikit-learn · Streamlit · Plotly

## Pistes d'amélioration

- Ajouter des modèles de gradient boosting (XGBoost, LightGBM).
- Tester des techniques de rééchantillonnage (SMOTE) et comparer avec la pondération des classes.
- Choisir les seuils selon le coût métier d'une fraude non détectée et d'un faux positif.
- Expliquer chaque décision individuelle avec SHAP.
- Exposer le modèle sous forme d'API pour l'intégrer à un système de paiement.

## Auteur

Franck Yanick KIANGUEBENI — Data Analyst, étudiant en Master Intelligence Artificielle (Dakar Institute of Technology).
