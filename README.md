# Une mouche qui joue à Tetris

Un réseau de neurones contraint par **le câblage mesuré du système nerveux
central d'une drosophile mâle** — reconstruit au microscope électronique —, et
entraîné à jouer à Tetris. La topologie ne bouge jamais : seules les forces des
connexions sont ajustées.

```
connectome MaleCNS v1.0 · 165 122 neurones tracés · 25 563 197 connexions
25 728 320 paramètres appris · 0 connexion ajoutée, retirée ou déplacée
```

## Voir la mouche jouer

La page `web/index.html` s'ouvre directement dans un navigateur, sans serveur.
Elle rejoue une partie enregistrée et montre, pour chaque pièce :

- **la grille**, avec les trois placements les mieux notés et leurs probabilités ;
- **le système nerveux pendant la décision** : 8 000 neurones du MaleCNS, chacun
  à la position de son soma — le cerveau en haut, la corde nerveuse ventrale en
  dessous. La vague d'activité part des neurones qui lisent la grille, envahit le
  cerveau en sept mises à jour, et atteint les neurones moteurs où la note est lue ;
- **la mouche sur sa manette**, dont les pattes appuient sur les touches — une
  suite reconstituée après coup à partir du placement choisi.

Pour la voir jouer **en direct** sur la carte graphique (il faut le connectome
préparé et un modèle entraîné) :

```bash
mouche diffuser --modele data/points_de_controle/<modèle>.pt   # puis http://localhost:8000
```

---

## La question

Beaucoup de projets branchent un connectome sur une tâche et montrent que ça
marche. Un réseau à 25 millions de paramètres entraînables apprend *toujours*
quelque chose — la démonstration ne prouve rien sur le système nerveux de la
mouche.

Ce projet pose une question mesurable :

> À recette identique, le câblage mesuré note-t-il mieux les grilles de Tetris
> qu'un câblage recâblé au hasard ?

Tout le reste — le moteur, les modèles de comparaison, les témoins, l'évaluation
appariée — existe pour que cette réponse veuille dire quelque chose.

## Résultats

### Lignes par partie

100 parties, graines 1000 à 1099, plafond 500 pièces, **mêmes séquences pour
toutes les conditions**. Intervalles à 95 % par bootstrap apparié.

| Condition | Lignes par partie | Paramètres |
|---|---|---|
| Hasard | 0,1 [0,0 ; 0,1] | — |
| Graphe coupé · entrée aveuglée | *à mesurer — notes égales vérifiées* | — |
| Juge de grilles linéaire | **32,7** [30,4 ; 35,1] | 206 |
| **Mouche** | *run complet en cours* | 25 728 320 |
| **Mouche recâblée** | *à mesurer* | 25 728 320 |
| Expert (heuristique à 4 critères) | **195,7** [195,2 ; 196,2] | — |

L'expert atteint le plafond dans 100 % des parties : ses 195,7 lignes sont un
**plancher**, pas une performance.

### Accord avec l'expert, en cours de route

La part des situations de test où le modèle choisit un placement que l'expert
juge optimal. C'est la mesure qui guide l'entraînement, pas le but — le but, ce
sont les lignes.

| Modèle | Accord | Mesuré sur |
|---|---|---|
| Hasard | 6,4 % | échantillon de 500 situations |
| Mouche, 200 pas de clonage (K = 7) | 46,6 % | échantillon de 500 |
| Mouche, 800 pas, recette d'origine | 53,9 % | jeu de test complet, 5 998 situations |
| **Mouche, 800 pas, recette corrigée** | **57,2 %** | jeu de test complet |
| Juge linéaire, à la fin de son clonage | 64,6 % | échantillon de 500 |

**À ce stade, la mouche n'atteint pas l'accord du juge linéaire** : 60,0 % sur
le même échantillon, contre 64,6 %. Son modèle n'a vu que 800 pas de clonage,
sans DAgger ; le run complet — 2 400 pas puis trois tours de DAgger — dira si
l'écart se referme. S'il ne se referme pas, ce sera publié tel quel.

Le détail, les échecs et les intervalles sont dans
[docs/resultats.md](docs/resultats.md) et [docs/conception.md](docs/conception.md) §10.3.

## Comment ça marche

La mouche ne choisit jamais une action. Le programme énumère les placements
possibles de la pièce courante — au plus 34 — et calcule la grille qui en
résulterait. Le réseau **note chaque grille** ; la mieux notée est jouée.

```
grille + pièce → moteur : liste des candidats (≤ 34)
  → encodage 205 entrées (200 cases après pose et disparition, +1/−1 ; 5 cases « lignes complétées »)
  → projection figée : chaque entrée lue par 20-21 des 4 114 neurones du lobe optique, signe aléatoire
  → K = 7 mises à jour du graphe, état remis à zéro pour chaque candidat
        état ← (1 − fuite) × état + fuite × tanh(somme des entrées)
  → note = Σ (708 neurones moteurs × coefficient figé) × température
  → argmax
```

### Ce qui s'entraîne, et ce qui ne s'entraîne jamais

| Figé | Appris |
|---|---|
| La topologie — qui parle à qui | Un gain par connexion (25 563 197) |
| Les signes (loi de Dale) | Une fuite par neurone (165 122) |
| Le nombre de synapses et sa normalisation | Une température (1) |
| Le facteur global, la projection d'entrée, le vecteur de sortie | |

Les gains passent par une softplus : **aucun ne peut devenir négatif**, donc
aucun signe ne peut s'inverser. Les 4 114 neurones d'entrée et les 708 neurones
moteurs sont **disjoints** : le seul chemin entre la grille et la note passe par
le câblage mesuré.

### La recette d'entraînement

Clonage de comportement sur 410 parties de l'expert, puis trois tours de DAgger
où la mouche joue seule et l'expert corrige les situations qu'elle a atteintes.

| Réglage | Valeur |
|---|---|
| Clonage | 2 400 pas, lots de 4 situations × 10 candidats, Adam |
| Perte | − log de la probabilité donnée aux placements optimaux de l'expert |
| Taux d'apprentissage | 0,04, décroissant en cosinus jusqu'à 0,004 |
| DAgger | 3 tours de 30 parties et 500 pas, lots moitié DAgger, moitié démonstrations |

Chaque changement de recette est d'abord essayé **à budget égal** contre la
recette en place, et mesuré sur tout le jeu de test avec un intervalle tiré par
parties entières (`mouche comparer-recettes`) :

| Essai, 800 pas | Écart d'accord (IC 95 %) | Décision |
|---|---|---|
| Perte sur les optimaux + taux décroissant | **+3,3 points** [+2,1 ; +4,3] | adoptée |
| + concurrents « difficiles », choisis par le modèle | +0,3 point [−1,2 ; +1,7] | non retenue |

Le run complet dure environ 7 h 30 et se découpe en séances ; la reprise restaure
l'état exact de l'optimiseur et du générateur, et elle est vérifiée bit pour bit
sur CPU.

## Les contrôles, qui font la différence

| Contrôle | Ce qu'il détecte |
|---|---|
| **Mouche recâblée** | Mêmes neurones, mêmes degrés, mêmes poids — mais qui-parle-à-qui détruit. Si elle fait aussi bien, le câblage précis n'apporte rien |
| **Graphe coupé** (gains effectifs à 0) | Un raccourci qui contournerait le connectome |
| **Entrée aveuglée** (entrées sensorielles à 0) | Un chemin par lequel la grille entrerait autrement |
| **Juge linéaire** (206 paramètres) | Que 25,7 millions de paramètres ne fassent pas moins bien que 206 |

Les deux témoins doivent retomber au niveau du hasard. S'ils font mieux, rien de
ce qui précède n'est interprétable.

## Démarrer

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[connectome,serveur,dev]"

pytest                    # 355 tests
pytest -m "not lent"      # sans les entraînements de plusieurs minutes
```

Les tests tournent **sans GPU et sans télécharger le connectome** : des graphes
de 20 et 1 200 neurones sont construits en mémoire. C'est une contrainte de
conception, pas une commodité.

```bash
mouche telecharger          # les 3 fichiers MaleCNS (1,2 Go), et rien d'autre
mouche preparer-graphe      # construit le graphe, vérifie 165 122 / 25 563 197
mouche demonstrations       # 410 parties de l'expert, ~101 000 situations
mouche reperes              # hasard et expert, protocole final
mouche juge-lineaire        # clonage + DAgger + évaluation
mouche run-court            # mise au point sur le connectome, 3 valeurs de K
mouche entrainer-mouche     # le run complet de la mouche : clonage + 3 tours de DAgger
mouche avancement           # où en est le run, sans toucher à la carte
mouche comparer-recettes    # un essai de recette, à budget égal
mouche enregistrer --modele M   # une partie de M, avec l'activité de ses neurones
mouche page                 # met cette partie dans web/index.html
mouche diffuser --modele M  # M joue en direct sur http://localhost:8000
```

Le run se découpe en séances :

```bash
mouche entrainer-mouche --duree 3 --perte optimaux --taux-final 0.004   # 3 h, puis rend la carte
mouche entrainer-mouche --duree 3   # reprend exactement là, dans la même recette
```

Les réglages par défaut du code sont encore ceux de la recette d'origine : la
recette adoptée se demande explicitement, comme ci-dessus. Une séance commencée
enregistre sa recette et refuse d'en changer en route.

### Matériel

Tout tourne sur une **RTX 3050 Mobile de 4 Go**, sans GPU loué.

| | Mesuré |
|---|---|
| VRAM à l'entraînement (40 évaluations, K = 7) | 2,15 Gio |
| VRAM en jeu (34 candidats en parallèle) | 1,36 Gio |
| Une pièce en jeu | 0,20 s |
| Un pas d'entraînement | 5,2 s |
| Un run complet (clonage + 3 tours de DAgger) | ~7 h 30 |

Le produit synaptique passe par le noyau creux de
[flyhard](https://github.com/MarkUnthank/flyhard) avec le correctif de
[fly-self-driving](https://github.com/suanmiao/fly-self-driving). Une
implémentation directe dépasse les 4 Go dès **deux** évaluations, et
`torch.sparse.mm` alloue un intermédiaire dense de **101,57 Gio** pour
rétropropager.

Sur un portable, une séance bloque la mise en veille **et** la fermeture du
capot, un chien de garde arrête un pas bloqué, et l'entraînement refuse de
tourner sans CUDA : une mise en veille coupe CUDA sans lever d'erreur
(`docs/conception.md` §14.6).

## Structure

| Chemin | Rôle |
|---|---|
| [`docs/conception.md`](docs/conception.md) | **Le document qui fait foi.** Toutes les décisions, tous les réglages |
| [`docs/resultats.md`](docs/resultats.md) | Résultats, intervalles, échecs |
| `src/mouche_tetris/graines.py` | Les familles de graines — seul endroit qui crée un générateur |
| `src/mouche_tetris/notation.py` | L'interface `Noteur`, commune à toutes les conditions |
| `src/mouche_tetris/encodage.py` | Les 205 entrées — seule frontière entre moteur et modèles |
| `src/mouche_tetris/tetris/` | Le moteur, unique moteur du projet, et la reconstitution des touches |
| `src/mouche_tetris/connectome/` | Graphe, signes, noyau creux, modèle, contrôles, nuage de neurones |
| `src/mouche_tetris/entrainement/` | Démonstrations, clonage, DAgger, séances |
| `src/mouche_tetris/evaluation/` | Parties appariées, bootstrap, accord avec l'expert |
| `src/mouche_tetris/enregistrement/` | Parties enregistrées pour la page, maillage de la mouche |
| `src/mouche_tetris/serveur/` | Le direct, et le verrou qui partage la carte avec l'entraînement |
| `web/` | La page : grille, nuage de neurones, mouche 3D |

Trois règles de dépendance tiennent l'architecture : le moteur n'importe jamais
PyTorch, toutes les conditions passent par la même interface, et aucun module ne
construit son propre générateur aléatoire.

## Ce que ce projet ne dit pas

Ce n'est **pas** une vraie mouche, **pas** un cerveau téléversé, et elle ne
comprend **pas** Tetris. La formulation exacte est :

> un réseau contraint par le câblage mesuré du système nerveux central d'une
> drosophile mâle, dont seules les forces des connexions ont été ajustées,
> complète X lignes par partie en moyenne.

Aucune fonction connue du système nerveux de la mouche ne correspond à Tetris :
la grille est envoyée à des neurones visuels par convention, ce n'est pas un
modèle d'œil. Le programme énumère les placements ; la mouche ne fait que juger
des grilles.

Sur la page, **le nuage de neurones montre l'état du modèle**, pas une activité
mesurée sur une mouche. Les neurones sensoriels du lobe optique n'ont pas de
soma dans le volume imagé — il est dans l'œil — : ils sont placés au centre de
leurs cibles. **La manette est une image** : la mouche n'appuie sur aucun
bouton, et la suite de touches est reconstituée après coup. La page dit les deux
à l'écran.

**Les échecs sont publiés comme les réussites.** Un résultat « le câblage précis
n'apporte rien à Tetris » est un résultat valable.

## Données et licences

Connectome **MaleCNS v1.0**, licence CC-BY 4.0 —
[male-cns.janelia.org](https://male-cns.janelia.org/download/). FlyEM (HHMI
Janelia), Université de Cambridge, MRC LMB, Google Research.

Maillage de la mouche : **NeuroMechFly** (NeLy, EPFL), Apache-2.0, issu d'un scan
micro-CT.

Code sous licence **MIT**. Les fichiers dérivés du MaleCNS — graphe préparé,
positions de soma, enregistrements d'activité — restent sous CC-BY 4.0.

Méthode reprise de [flyhard](https://github.com/MarkUnthank/flyhard) (Mark
Unthank) et [fly-self-driving](https://github.com/suanmiao/fly-self-driving)
(Pure Reason Inc.), tous deux MIT.

Critères de l'expert : l'IA de Tetris de Yiyuan Lee. DAgger : Ross, Gordon et
Bagnell, 2011.
