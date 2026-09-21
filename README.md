# Une mouche qui joue à Tetris

Un réseau de neurones dont **le câblage est celui d'une vraie drosophile**, mesuré
au microscope électronique, entraîné à jouer à Tetris. La topologie ne bouge
jamais : seules les forces des connexions sont ajustées.

```
connectome MaleCNS v1.0 · 165 122 neurones tracés · 25 563 197 connexions
25 728 320 paramètres appris · 0 connexion ajoutée, retirée ou déplacée
```

---

## La question

Beaucoup de projets branchent un connectome sur une tâche et montrent que ça
marche. Un réseau à 25 millions de paramètres entraînables apprend *toujours*
quelque chose — la démonstration ne prouve rien sur le cerveau de la mouche.

Ce projet pose une question mesurable :

> À recette identique, le câblage mesuré note-t-il mieux les grilles de Tetris
> qu'un câblage recâblé au hasard ?

Tout le reste — le moteur, les modèles de comparaison, les témoins, l'évaluation
appariée — existe pour que cette réponse veuille dire quelque chose.

## Résultats

100 parties, graines 1000 à 1099, plafond 500 pièces, **mêmes séquences pour
toutes les conditions**. Intervalles à 95 % par bootstrap apparié.

| Condition | Lignes par partie | Paramètres |
|---|---|---|
| Hasard | 0,1 [0,0 ; 0,1] | — |
| Graphe coupé · entrée aveuglée | *attendu 0,1* | — |
| Juge de grilles linéaire | **32,7** [30,4 ; 35,1] | 206 |
| **Mouche** | *en cours* | 25 728 320 |
| **Mouche recâblée** | *à mesurer* | 25 728 320 |
| Expert (heuristique à 4 critères) | **195,7** [195,2 ; 196,2] | — |

L'expert atteint le plafond dans 100 % des parties : ses 195,7 lignes sont un
**plancher**, pas une performance.

**Premier signe de vie du connectome.** Après 200 mises à jour seulement, la
mouche choisit le même placement que l'expert dans **43,8 %** des situations de
test, contre 6,4 % au hasard.

Le détail, les échecs et les intervalles sont dans [docs/resultats.md](docs/resultats.md).

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

Cette formulation « juge de grilles » est le cœur de la conception : à taille
comparable, elle est donnée pour ~30 fois meilleure que le choix direct d'une
action.

### Ce qui s'entraîne, et ce qui ne s'entraîne jamais

| Figé | Appris |
|---|---|
| La topologie — qui parle à qui | Un gain par connexion (25 563 197) |
| Les signes (loi de Dale) | Une fuite par neurone (165 122) |
| Le nombre de synapses et sa normalisation | Une température (1) |
| Le facteur global, la projection d'entrée, le vecteur de sortie | |

Les gains passent par une softplus : **aucun ne peut devenir négatif**, donc
aucun signe ne peut s'inverser, quelle que soit la taille du pas d'apprentissage.

Les 4 114 neurones d'entrée et les 708 neurones moteurs sont **disjoints**, et
le seul chemin entre la grille et la note passe par le câblage mesuré.

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
pip install -e ".[connectome,dev]"

pytest                    # 265 tests, ~1 min
pytest -m "not lent"      # sans les entraînements, ~12 s
```

Les tests tournent **sans GPU et sans télécharger le connectome** : un graphe
jouet de 20 neurones est construit en mémoire. C'est une contrainte de
conception, pas une commodité.

```bash
mouche telecharger        # les 3 fichiers MaleCNS (1,2 Go), et rien d'autre
mouche preparer-graphe    # construit le graphe, vérifie 165 122 / 25 563 197
mouche demonstrations     # 410 parties de l'expert, ~101 000 situations
mouche reperes            # hasard et expert, protocole final
mouche juge-lineaire      # clonage + DAgger + évaluation
mouche run-court          # mise au point sur le connectome, 3 valeurs de K
mouche entrainer-mouche   # le run complet de la mouche : clonage + 3 tours de DAgger
mouche avancement         # où en est le run, sans toucher à la carte
mouche diffuser           # la mouche joue en direct sur http://localhost:8000
```

Le run complet dure environ 7 h 30. Il se découpe en séances, sans rien changer
au résultat — la reprise est vérifiée bit pour bit :

```bash
mouche entrainer-mouche --duree 3   # travaille 3 h, enregistre, rend la carte
mouche entrainer-mouche --duree 3   # reprend exactement là
mouche entrainer-mouche             # va jusqu'au bout
```

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

## Structure

| Chemin | Rôle |
|---|---|
| [`docs/conception.md`](docs/conception.md) | **Le document qui fait foi.** Toutes les décisions, tous les réglages |
| [`docs/resultats.md`](docs/resultats.md) | Résultats, intervalles, échecs |
| `src/mouche_tetris/graines.py` | Les trois familles de graines — seul endroit qui crée un générateur |
| `src/mouche_tetris/notation.py` | L'interface `Noteur`, commune à toutes les conditions |
| `src/mouche_tetris/encodage.py` | Les 205 entrées — seule frontière entre moteur et modèles |
| `src/mouche_tetris/tetris/` | Le moteur, unique moteur du projet |
| `src/mouche_tetris/connectome/` | Graphe, signes, noyau creux, modèle, contrôles |
| `src/mouche_tetris/entrainement/` | Démonstrations, clonage, DAgger, séances |
| `src/mouche_tetris/evaluation/` | Parties appariées, bootstrap, accord avec l'expert |
| `web/` | La page de replay |

Trois règles de dépendance tiennent l'architecture : le moteur n'importe jamais
PyTorch, toutes les conditions passent par la même interface, et aucun module ne
construit son propre générateur aléatoire.

## Ce que ce projet ne dit pas

Ce n'est **pas** une vraie mouche, **pas** un cerveau téléversé, et elle ne
comprend **pas** Tetris. La formulation exacte est :

> un réseau contraint par le câblage mesuré du système nerveux central d'une
> drosophile mâle, dont seules les forces des connexions ont été ajustées,
> complète X lignes par partie en moyenne.

Aucune fonction connue du cerveau de la mouche ne correspond à Tetris : la
grille est envoyée à des neurones visuels par convention, ce n'est pas un modèle
d'œil. Le programme énumère les placements ; la mouche ne fait que juger des
grilles.

Sur la démo, la manette est une image : la mouche n'appuie sur aucun bouton, et
la séquence de touches est reconstituée après coup. La page le dit à l'écran.

**Les échecs sont publiés comme les réussites.** Un résultat « le câblage précis
n'apporte rien à Tetris » est un résultat valable.

## Données et licences

Connectome **MaleCNS v1.0**, licence CC-BY 4.0 —
[male-cns.janelia.org](https://male-cns.janelia.org/download/). FlyEM (HHMI
Janelia), Université de Cambridge, MRC LMB, Google Research.

Code sous licence **MIT**. Les fichiers dérivés du MaleCNS — graphe préparé,
positions de soma, enregistrements d'activité — restent sous CC-BY 4.0.

Méthode reprise de [flyhard](https://github.com/MarkUnthank/flyhard) (Mark
Unthank) et [fly-self-driving](https://github.com/suanmiao/fly-self-driving)
(Pure Reason Inc.), tous deux MIT.

Critères de l'expert : l'IA de Tetris de Yiyuan Lee. DAgger : Ross, Gordon et
Bagnell, 2011.
