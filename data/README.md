# Description des données HumanForYou

Les données ont été anonymisées : chaque employé de l'entreprise sera représenté par le même EmployeeID dans l'ensemble des fichiers qui suivent.

### Données du service des ressources humaines - general_data.csv

- Age : L'âge de l'employé en 2015.
- Attrition : L'objet de notre étude, est-ce que l'employé a quitté l'entreprise durant l'année 2016 ?
- BusinessTravel : A quelle fréquence l'employé a-t-il été amené à se déplacer dans le cadre de son travail en 2015 ? (Non-Travel = jamais, Travel_Rarely= rarement, Travel_Frequently = fréquemment).
- DistanceFromHome : Distance en km entre le logement de l'employé et l'entreprise.
- Education : Niveau d'étude : 1=Avant College (équivalent niveau Bac), 2=College (équivalent Bac+2), 3=Bachelor (Bac+3), 4=Master (Bac+5) et 5=PhD (Thèse de doctorat).
- EducationField : Domaine d'étude, matière principale.
- EmployeeCount : booléen à 1 si l'employé était compté dans les effectifs en 2015.
- EmployeeId : l'identifiant d'un employé.
- Gender : Sexe de l'employé.
- JobLevel : Niveau hiérarchique dans l'entreprise de 1 à 5.
- JobRole : Métier dans l'entreprise.
- MaritalStatus : Statut marital du salarié (Célibataire, Marié ou Divorcé).
- MonthlyIncome : Salaire brut en roupies par mois.
- NumCompaniesWorked : Nombre d'entreprises pour lesquelles le salarié a travaillé avant de rejoindre HumanForYou.
- Over18 : Est-ce que le salarié a plus de 18 ans ou non ?
- PercentSalaryHike : % d'augmentation du salaire en 2015.
- StandardHours : Nombre d'heures par jour dans le contrat du salarié.
- StockOptionLevel : Niveau d'investissement en actions de l'entreprise par le salarié.
- TotalWorkingYears : Nombre d'années d'expérience en entreprise du salarié pour le même type de poste.
- TrainingTimesLastYear : Nombre de jours de formation en 2015.
- YearsAtCompany : Ancienneté dans l'entreprise.
- YearsSinceLastPromotion : Nombre d'années depuis la dernière augmentation individuelle.
- YearsWithCurrentManager : Nombre d'années de collaboration sous la responsabilité du manager actuel de l'employé.

### Dernière évaluation du manager - manager_survey_data.csv

Dernière évaluation de chaque employé faite par son manager en février 2015.

- L'identifiant de l'employé : EmployeeID
- Une évaluation de son implication dans son travail notée 1 ('Faible'), 2 ("Moyenne"), 3 ("Importante") ou 4 ("Très importante") : JobInvolvement
- Une évaluation de son niveau de performance annuel pour l'entreprise notée 1 ("Faible"), 2 ("Bon"), 3 ("Excellent") ou 4 ("Au delà des attentes") : PerformanceRating

### Enquête qualité de vie au travail - employee_survey_data.csv

Ce fichier provient d'une enquête soumise aux employés en juin 2015 par le service RH pour avoir un retour concernant leur qualité de vie au travail.

Lorsqu'un employé n'a pas répondu à une question, le texte "NA" apparaît à la place de la note.

- l'environnement de travail, noté 1 ("Faible"), 2 ("Moyen"), 3 ("Élevé") ou 4 ("Très élevé") : EnvironmentSatisfaction
- son travail, noté de 1 à 4 comme précédemment : JobSatisfaction
- son équilibre entre vie professionnelle et vie privée, noté 1 ("Mauvais"), 2 ("Satisfaisant"), 3 ("Très satisfaisant") ou 4 ("Excellent") : WorkLifeBalance

### Horaires de travail - in_time.csv - out_time.csv

Horaires d'entrée et de sortie des employés sur une période de l'année choisie, représentative d'une activité moyenne pour l'ensemble des services.