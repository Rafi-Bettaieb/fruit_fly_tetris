"""Démonstrations, clonage de comportement, DAgger (conception.md §10).

- `demonstrations`      — 410 parties de l'expert, graines 0–409, plafond 300,
                          5 % de coups au hasard avec l'étiquette experte,
                          soit environ 100 000 situations
- `clonage`             — softmax sur 10 candidats, lots de 4, Adam 0,04,
                          2 400 mises à jour
- `dagger`              — 3 tours de 30 parties + 500 mises à jour, lots tirés
                          **moitié** dans les données DAgger, moitié dans les
                          démonstrations (§10.4)
- `points_de_controle`  — toutes les 200 mises à jour ; paramètres **et** état
                          d'Adam **et** interfaces tirées, environ 0,31 Go

Deux règles opératoires à implémenter, pas à improviser :

- **divergence** : perte NaN, ou en hausse de plus de 50 % sur 100 mises à jour ⇒
  reprise du dernier point de contrôle avec un taux divisé par 4. **Trois
  relances au maximum**, ensuite le run est déclaré échoué et publié comme tel ;
- **choix du meilleur point de contrôle** : accord avec l'expert sur un
  sous-échantillon **fixe** de 500 situations de test. Pas sur les 6 000 : les
  noter toutes coûte une demi-heure de GPU, soit 6 heures par run.

Pourquoi 2 400 mises à jour et non 1 200 : un pas porte ici 4 termes de perte,
contre 64 dans la référence, qui rétropropageait à travers 16 décisions. À
nombre de pas égal, le signal de gradient serait 16 fois plus faible (§2.3).
"""
