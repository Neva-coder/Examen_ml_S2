Rapport de projet – Atlantic Haven Hotels
Examen Final Machine Learning & Data Science — M1

1.	Informations sur le Groupe
Membre 1
•	Nom : RAMIAKATRARIVO
•	Prénoms : Anjara Fifaliana Tendrin’Iavo 
•	Classe : IGGLIA 4
•	Numéro : 14
•	Rôle : Responsable modélisation ML
Membre 2
•	Nom : ANJARATINA NY AINA 
•	Prénoms : Antsa Christiana 
•	Classe : IGGLIA 4
•	Numéro : 29
•	Rôle : Data analyst
Membre 3
•	Nom : RAJAONARISON
•	Prénoms : Tsanta Emmanuël
•	Classe : IGGLIA 4
•	Numéro : 34
•	Rôle : Data engineering
Membre 4
•	Nom : RAKOTONARIVO
•	Prénoms : Nomenjanahary Faneva
•	Classe : IGGLIA 4
•	Numéro : 48
•	Rôle : Chef de projet et rédacteur technique
Membre 5
•	Nom : NANTENAINASOA
•	Prénoms : Oliva Fitahina
•	Classe : IGGLIA 4
•	Numéro : 61
•	Rôle : Développeur pipeline

2.	Résumé de travail
Problématique
Atlantic Haven Hotels fait face à un taux d'annulation important de ses réservations (~25,8%), engendrant une sous-optimisation du taux d'occupation de ses chambres et des pertes de chiffre d'affaires sèches. L'objectif est de prédire le risque d'annulation suffisamment tôt à l'aide du Machine Learning afin d'anticiper la réattribution des chambres et de mettre en place des politiques de surréservation ou de relance ciblée.

Méthodologie adoptée
1. « EDA & Nettoyage » : Imputation sans fuite de données (*data leakage*) des valeurs manquantes (`agent_id` imputé par 0, variables numériques imputées par la médiane du train, variables catégorielles par le mode du train).
2. « Feature Engineering » :  Création du délai d'anticipation de réservation (`delai_reservation`), ainsi que du mois (`mois_arrivee`) et du jour de la semaine (`jour_semaine_arrivee`).
3. « Validation Temporelle » :  Découpage chronologique strict (80% passé pour le Train = 6 400 lignes, 20% récent pour la Validation = 1 600 lignes) pour simuler les conditions réelles de production.
4. « Modélisation & Seuil » : Comparaison d'une régression logistique baseline avec des modèles gérant le déséquilibre des classes (Logistic Regression Pondérée, Random Forest, Gradient Boosting / XGBoost) et recherche du seuil de décision optimal pour maximiser le F1-score sur la classe minoritaire.

Résultats obtenus
Le modèle final XGBoost / Gradient Boosting combiné à un seuil de décision ajusté à 0.32 permet d'atteindre un F1-score de 0.4686 (contre 0.1464 pour la baseline initiale avec un seuil par défaut de 0.50). Il permet de capturer plus de 78,3% des annulations réelles (Rappel = 0.7832).

Mots clés
Classification binaire, Annulation hôtelière, Validation temporelle, F1-score, Feature engineering, Déséquilibre de classes, XGBoost.

3.	Contenu du repository
├── notebook.ipynb
├── submission.csv
├── README.md
└── requirements.txt
Liens utiles : 
•	Lien vidéo de présentation : 
•	Lien dépôt GitHub : https://github.com/Neva-coder/Examen_ml_S2
4.	Résultat et Modélisation
Présentation des résultats obtenus : 







Modèle	Paramètres principaux	F1-score	Précision	Rappel	ROC-AUC
Régression logistique —baseline	Seuil par défaut = 0.50	0.1464	0.4222	0.0886	0.6478
Régression logistique — équilibrée	`class_weight='balanced'`, Seuil = 0.40	0.4652	0.3344	0.7646	0.6452
Random Forest	`n_estimators=200`, `class_weight='balanced'`, Seuil = 0.20	0.4591	0.3255	0.7786	0.6314
Modèle final (XGBoost / Gradient Boosting)	`scale_pos_weight`, `learning_rate=0.05`, `max_depth=6`, Seuil = 0.32	0.4686	0.3343	0.7832	0.6403

Seuil de décision retenu : 

Justification du choix du modèle final :
Le modèle « XGBoost / Gradient Boosting » surpasse la baseline et les autres algorithmes en termes de F1-score et de Rappel. L'abaissement du seuil à 0.32 permet au modèle de devenir plus vigilant face à la classe minoritaire des annulations, capturant ainsi **78,3% des annulations réelles** au prix d'une baisse modérée de précision, ce qui est parfaitement aligné avec l'objectif financier de l'hôtel (éviter les chambres vides un jour J).

5.	 Réponses aux Questions d’Analyse
Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l’accuracy pour cette tâche ?
L'Accuracy est trompeuse sur ce dataset car les classes sont déséquilibrées (~25,8% d'annulations pour ~74,2% de réservations maintenues). Un modèle naïf ou paresseux qui prédirait que personne n'annule obtiendrait une Accuracy de 74,2%, tout en étant totalement inutile pour l'hôtel (0% des annulations détectées). Le F1-score combine la Précision et le Rappel sous forme de moyenne harmonique et exige du modèle qu'il soit réellement performant sur la classe minoritaire d'intérêt (`reservation_annulee = 1`).

Q2. Dans ce contexte, qu’est-ce qui est le plus grave : un faux positif ou un faux négatif ?
Dans le contexte hôtelier, un « faux négatif » (prédire qu'un client va venir alors qu'il annule) est généralement plus grave qu'un faux positif. 
- Impact d'un Faux Négatif :  L'hôtel ne prend aucune mesure et se retrouve avec une chambre inoccupée au dernier moment sans possibilité de relouer, représentant une perte sèche de revenu.
- Impact d'un Faux Positif : L'hôtel anticipe à tort une annulation et déclenche une action de relance (confirmation par SMS/mail, offre de surclassement) ou autorise un léger overbooking. Si le client vient tout de même, le coût de gestion est minime. 

Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?
La variable la plus contributive créée est `delai_reservation` (calculée en effectuant la différence en jours entre `date_arrivee` et `date_reservation`).
- Justification : Le délai d'anticipation varie de 0 à plus de 580 jours dans le dataset. Les données montrent que le taux d'annulation passe de 23,1% pour les réservations faites à moins de 14 jours de l'arrivée à plus de 33,1% pour les réservations effectuées plus de 56 jours à l'avance.
- Gain observé : L'ajout de cette variable combiné à la décomposition saisonnière (`mois_arrivee`, `jour_semaine_arrivee`) a permis de faire bondir le Rappel du modèle de 8,8% à plus de 78%.

Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?
Un découpage aléatoire classique (K-Fold ou train_test_split aléatoire) détruit la structure temporelle des données et provoque une fuite d'information temporelle (data leakage). Dans la réalité, l'hôtel doit prédire les annulations futures à partir de données passées.
Pour refléter cette contrainte, nous avons appliqué un découpage chronologique strict : tri des données par `date_reservation`, utilisation des 80% plus anciennes réservations (6 400 lignes) pour l'entraînement et des 20% plus récentes (1 600 lignes) pour la validation.

Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?
1. **Les réservations très anticipées sans demandes particulières : ** Un délai de réservation supérieur à 56 jours combiné à `demandes_speciales = 0` présente le taux d'annulation le plus élevé (33,1%).
2. **Les réservations à prix élevé sur les marchés européens : ** Les réservations avec un prix moyen par nuit supérieur à 230 € et provenant des marchés *Europe_Centrale* ou *Italie* présentent un risque d'annulation supérieur à 27%.
3. **Le manque d'implication du client :** Les clients n'ayant émis aucune demande spécifique (`demandes_speciales = 0`) annulent dans 28,2% des cas, contre seulement 20,2% pour les clients ayant fait 3 demandes spécifiques ou plus.

Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l’entraînement ?
* **Valeurs manquantes : ** Traitées sans fuite de données (*data leakage*). `agent_id` est imputé par `0` puis converti en type texte (`str`). Les variables numériques (`enfants`, `prix_moyen_nuit_eur`, `demandes_speciales`) sont imputées par la **médiane** calculée uniquement sur le train set. Les variables catégorielles (`marche_origine`) sont imputées par le **mode** du train set.
* **Catégories inédites dans le test set : ** Le préprocesseur utilise `OneHotEncoder(handle_unknown='ignore’) `. Si une catégorie inconnue apparaît lors du test, l'encodeur lui attribue un vecteur constitué uniquement de zéros sans bloquer l'exécution.

Q7. Selon vous, quelle action l’hôtel devrait-il entreprendre lorsqu’une réservation en cours présente une forte probabilité d’annulation ?
L'hôtel ne doit **jamais annuler d'office** la réservation d'un client. Les actions recommandées sont :
1. **Relance préventive à J-7 : ** Envoyer un message automatique (SMS/E-mail) demandant au client de reconfirmer sa venue en échange d'un avantage (ex : petit-déjeuner offert ou enregistrement anticipé).
2. **Sur-réservation contrôlée (Overbooking) : ** Autoriser un taux d'overbooking mesuré (ex : 5% à 10%) uniquement sur les types de chambres où le modèle détecte une forte concentration d'annulations probables.

Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?
Non, l'évaluation par région montre une variabilité des performances due à la taille des sous-échantillons et aux spécificités touristiques locales :
- « Sicile (`Sicilia`) » : F1-score de **0.5802** (121 réservations).
- « Lombardie (`Lombardia`) » : F1-score de **0.5355** (216 réservations).
- « Ligurie (`Liguria`) » : F1-score de **0.3871** (127 réservations).
Discussion des limites :  Sur les régions disposant d'un faible volume de données en validation (ex: Sardaigne avec 67 réservations), la métrique est très sensible aux variations individuelles.

Q9. Analyse des erreurs
5 Faux Positifs (Prédit : Annulé (1), Réalité : Maintenu (0)) :
•	**R000308** (Probabilité : 0.4273) : Réservation faite 45 jours à l'avance avec un prix de 135.80 €.
•	**R001943** (Probabilité : 0.5765) : Délai de 43 jours et tarif élevé de 245.31 €.
•	**R009259** (Probabilité : 0.5965) : Prix très élevé (223.70 €) ayant déclenché une alerte malgré 2 demandes spéciales.
•	**R008402** (Probabilité : 0.3318) : Absence totale de demandes spéciales (`demandes_speciales = 0`) et délai de 25 jours.
•	**R007351** (Probabilité : 0.5622) : Profil tarifaire élevé (175.98 €) interprété comme à risque par le modèle.

5 Faux Négatifs (Prédit : Maintenu (0), Réalité : Annulé (1)) :
•	**R004804** (Probabilité : 0.2675) : Réservation faite au dernier moment (délai de 2 jours) mais annulée subitement.
•	**R005322** (Probabilité : 0.3127) : Délai court (21 jours) et prix modéré (168.46 €) ayant induit le modèle en erreur.
•	**R006059** (Probabilité : 0.2481) : Présence de 2 demandes spéciales qui a faussement rassurer le modèle.
•	**R003178** (Probabilité : 0.2427) : Délai d'anticipation très court (13 jours) suivi d'une annulation imprévue.
•	**R002592** (Probabilité : 0.1714) : Faible tarif (120.74 €) ayant masqué le risque d'annulation.

Raisons des erreurs & Piste d'amélioration :
•	**Raisons : ** Le jeu de données ne contient pas d'informations sur les motifs imprévisibles du client (météo, urgences personnelles, annulations de vols).
•	**Piste d'amélioration : ** Intégrer l'historique d'annulation personnel du client (*remboursements passés*) ainsi que des données externes (prévisions météo régionales, événements locaux).

6.	Conclusion et Recommandations
Le modèle développé (XGBoost avec ajustement de seuil à 0.32) permet de corriger le défaut de la baseline initiale en capturant **78,3% des annulations réelles**. Malgré une précision modérée liée au faible nombre de caractéristiques comportementales dans le dataset, le système apporte une valeur opérationnelle directe.

**Recommandation opérationnelle finale : **
Déployer le modèle pour alimenter un tableau de bord hebdomadaire. Utiliser les probabilités fournies pour déclencher des alertes automatiques de confirmation à J-7 et piloter une politique de sur-réservation prudente (max 5-8%) sur les périodes identifiées comme critiques.


7.	Reproductibilité
Version de pyrhon : Python 3.12 (ou 3.10)
Principales bibliothèques et versions : `pandas==2.2.2`, `numpy==1.26.4`, `scikit-learn==1.4.2`, `xgboost==2.0.3`
Graines aléatoires : `random_state = 42`
Commande ou procédure d’exécution : Exécuter la commande `python main.py` ou lancer l'intégralité du notebook `notebook.ipynb`.
Durée approximative d’entrainement : ~ 15 secondes
Environnement utilisé : Google Colab / Environnement local Python

8.	Bibliographie
•	Documentation Scikit-Learn (*ColumnTransformer, Pipeline, LogisticRegression*) : https://scikit-learn.org/
•	Documentation XGBoost Python API : https://xgboost.readthedocs.io/
•	Documentation Pandas (*Time series / Datetime properties*) : https://pandas.pydata.org/
•	Outils d'IA générative :  Assistance apportée par l'IA Gemini pour le débogage des erreurs d'encodage `int/str` et le calcul optimisé du seuil de décision.
