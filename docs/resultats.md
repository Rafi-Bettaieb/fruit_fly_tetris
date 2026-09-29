# Résultats

Toutes les mesures de ce fichier sont produites par le dépôt, protocole final de
`conception.md` §11.1 : **100 parties, graines 1000 à 1099, plafond 500 pièces**,
mêmes séquences pour toutes les conditions. Intervalles à 95 % par bootstrap
apparié, 10 000 tirages, graine 0.

Les échecs et les résultats négatifs figurent ici au même titre que les autres.

## État

Les deux bornes et le juge linéaire sont mesurés. Le MaleCNS est préparé et le
connectome apprend (mise au point ci-dessous) ; les essais de recette, faits à
800 pas et mesurés sur tout le jeu de test, sont dans `conception.md` §10.3. Le
run complet de la mouche, la mouche recâblée et les deux témoins restent à
mesurer sur le protocole final.

## Tableau

| Condition | Lignes (IC 95 %) | Médiane | Pièces | Plafond | Trous/pièce | Budget |
| --- | --- | --- | --- | --- | --- | --- |
| hasard | 0,1 [0,0 ; 0,1] | 0 | 25,0 | 0 % | 3,624 | aucun |
| graphe coupé | *à mesurer* — notes égales vérifiées | | | | | |
| entrée aveuglée | *à mesurer* — notes égales vérifiées | | | | | |
| **juge linéaire** | **32,7 [30,4 ; 35,1]** | **32** | **123,9** | **0 %** | **0,350** | **29 900 pas · 206 paramètres** |
| mouche recâblée | *à mesurer* | | | | | |
| mouche | *à mesurer* | | | | | 25 728 320 paramètres |
| expert | 195,7 [195,2 ; 196,2] | 197 | 500,0 | 100 % | 0,111 | aucun — 4 poids figés |

**Le juge linéaire place la barre deux fois et demie plus haut que prévu.** Le
document attendait 13,8 lignes ; il en fait 32,7 avec 206 paramètres. C'est donc
ce chiffre-là que la mouche doit dépasser, et non celui de la conception.

Son accord avec l'expert passe de 6,4 % — le niveau du hasard — à 64,6 %.

### DAgger n'a rien apporté au juge linéaire

| Étape | Lignes (protocole de développement) |
| --- | --- |
| Après clonage | 32,6 |
| Après 3 tours de DAgger | 30,3 |

Le document attendait un doublement. Trois explications restent ouvertes, et
rien ici ne permet de les départager : un modèle à 206 paramètres était
peut-être déjà à son plafond ; le rééquilibrage moitié-moitié des lots change la
donne par rapport aux mesures citées ; ou 4 700 mises à jour par tour sont trop
peu.

**C'est un résultat, pas une anomalie à corriger.** Il ne préjuge pas de ce que
DAgger fera pour la mouche, dont la capacité est d'un tout autre ordre.

**Lecture.** L'expert atteint le plafond dans 100 % des parties : ses 195,7 lignes
sont donc un **plancher**, pas une performance. Avec un plafond de 500 pièces,
le maximum atteignable est 200 lignes.

La marge attendue est là : du hasard à l'expert, de 0,1 à 195,7 lignes, il y a
la place nécessaire pour que l'effet du câblage soit mesurable — c'est ce qui
manquait au projet Uno.

## Écarts appariés

| Écart | Différence (IC 95 %) | Conclusion |
| --- | --- | --- |
| expert − hasard | 195,7 [195,1 ; 196,2] | écart démontré |

**Note de méthode.** Sur cette paire-là, l'appariement n'apporte presque rien :
l'expert est tronqué par le plafond, donc sa variance est déjà minuscule. Il
prendra tout son sens sur la comparaison qui porte la question du projet — la
mouche contre la mouche recâblée — où les deux conditions affrontent les mêmes
séquences et où l'écart attendu est petit.

## Validation de la chaîne, sur un connectome réduit

Avant l'arrivée du MaleCNS, l'enchaînement complet est vérifié sur un connectome
construit de 1 200 neurones et 11 193 connexions — encodage, projection d'entrée
figée, K mises à jour du graphe, lecture motrice, perte softmax, rétropropagation
à travers les K mises à jour, jeu.

| Mesure | Valeur |
| --- | --- |
| Distance D (entrée → 90 % des moteurs) | 2 |
| Facteur global retenu | 1,0 — 85 % de moteurs actifs, 0 % de saturés |
| Accord avec l'expert : hasard → départ → après 600 pas | 6,9 % → 9,3 % → **57,7 %** |
| Participation, avant et après entraînement | 51 % → 62 % |
| Jeu | **3,5 lignes, 37 pièces** (hasard : 0,1 ligne, 25 pièces) |

Un réseau dont la topologie est imposée, et dont seules les forces ont été
ajustées, joue donc à Tetris mieux que le hasard. **Ce n'est pas un résultat
scientifique** — le graphe est construit, pas mesuré, et 1 200 neurones ne
feront jamais 195 lignes. C'est la preuve que la chaîne tient debout avant
d'engager des heures de GPU sur le connectome réel.

Note au passage : le graphe jouet de 20 neurones prescrit pour l'intégration
continue **ne peut pas** servir à cela. Avec 4 neurones sensoriels pour 205
entrées, il ne lit que 4 cases de la grille. Il reste le support des tests de
structure — orientation, loi de Dale, normalisation — où il est parfait.

## Mise au point sur le connectome complet

200 mises à jour de clonage, trois valeurs du temps de réflexion K, sur les
165 122 neurones et 25 563 197 connexions du MaleCNS. Accord mesuré sur le
sous-échantillon fixe de 500 situations de test.

| K | Accord au départ | Après 200 pas | Gain |
|---|---|---|---|
| D + 2 = 5 | 6,4 % | 43,8 % | +37,4 |
| **D + 4 = 7** | 9,6 % | **46,6 %** | +37,0 |
| D + 8 = 11 | 11,2 % | 44,0 % | +32,8 |

**Le connectome apprend.** En 200 mises à jour, la mouche choisit le même
placement que l'expert dans **46,6 %** des situations, contre **6,4 %** au
hasard. Le seuil du point d'arrêt était à 10 %.

**K = D + 4 est confirmé par la mesure.** C'était la règle la plus arbitraire du
document ; elle donne le meilleur des trois. Et K = 11 fait *moins bien* que
K = 7 : allonger le temps de réflexion ne rend pas la mouche meilleure.

Nuance à citer avec ces chiffres : les points de départ diffèrent. Avant tout
apprentissage, le réseau est déjà au-dessus du hasard à K élevé, et une partie
de l'écart entre les colonnes ne vient donc pas de l'apprentissage. Mesuré en
gain, K = 5 et K = 7 sont à égalité et K = 11 décroche seul.

## Les témoins d'intégrité

| Témoin | Notes toutes égales | Accord |
|---|---|---|
| Graphe coupé — gains effectifs à 0 | **oui** | 6,0 % |
| Entrée aveuglée — entrées sensorielles à 0 | **oui** | 6,0 % |

Les deux rendent une note **strictement identique** à tous les candidats : le
choix redevient purement aléatoire, et l'accord retombe au niveau du hasard.

C'est la vérification la plus importante du projet. Sans elle, les 46,6 %
pourraient venir d'un chemin qui contournerait le connectome — une régularité de
l'énumération, une fuite entre candidats dans le noyau creux, un biais de la
projection d'entrée. Les deux témoins ferment ces portes : **tout le signal
passe bien par le câblage mesuré.**

## Ce que ces intervalles ne mesurent pas

Ils mesurent le bruit de partie à partie, à graine de run fixée. Ils ne disent
rien de la variabilité d'une graine à l'autre, celle du tirage des interfaces et
de l'optimisation. Avec trois graines, cette seconde variabilité sera vue, pas
estimée : le facteur limitant des conclusions sera le nombre de graines, pas le
nombre de parties.

## Reproduire

```bash
pip install -e ".[dev]"
pytest
```
