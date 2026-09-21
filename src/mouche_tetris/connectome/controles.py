"""Les contrôles : ce qui transforme une démo en mesure (conception.md §11.2).

Un réseau de 25,7 millions de paramètres **apprendra quelque chose**, quelle que
soit sa topologie. Si la mouche fait quarante lignes, cela ne dit rien du cerveau
de la drosophile — cela dit qu'on a entraîné un gros modèle. Ce sont ces trois
conditions qui font la différence entre une démonstration et un résultat.

- **mouche recâblée** : la question du projet. Mêmes neurones, mêmes degrés
  entrants, mêmes poids de base ; seul qui-parle-à-qui est détruit.
- **graphe coupé** et **entrée aveuglée** : les témoins d'intégrité. Ils doivent
  retomber au niveau du hasard, environ 0,1 ligne. S'ils font mieux, un
  raccourci transporte le signal hors du connectome et **rien** de ce qui
  précède n'est interprétable.
"""

from __future__ import annotations

import numpy as np

from .graphe import Graphe
from .modele import Mouche


def recabler(graphe: Graphe, graine: int = 0) -> Graphe:
    """Permute les indices présynaptiques sur toutes les connexions (§11.2).

    Ce que la permutation **conserve** : pour chaque neurone postsynaptique, son
    degré entrant et le multiensemble des nombres de synapses qu'il reçoit. La
    normalisation de §7.4 est donc inchangée, et les deux mouches ont exactement
    les mêmes poids de base en valeur absolue. Le nombre total de connexions ne
    bouge pas non plus.

    Ce qu'elle **détruit** : qui parle à qui, et les degrés sortants, qui
    deviennent à peu près poissonniens.

    Ce qu'elle détruit **en plus, sans le vouloir** : comme le signe suit le
    nouveau neurone présynaptique, l'équilibre excitation / inhibition reçu par
    chaque neurone se rapproche de la moyenne du réseau. La mouche recâblée perd
    donc deux choses à la fois — le câblage précis *et* l'équilibre local des
    signes. C'est la limite de ce contrôle, et elle doit figurer dans les
    résultats : un écart mesuré ne dit pas laquelle des deux comptait.

    Les boucles sur soi et les doublons créés par le tirage sont **conservés**,
    pour que le nombre de connexions reste exactement celui du graphe mesuré.
    Leur nombre est rapporté par `anomalies_du_recablage`.
    """
    generateur = np.random.default_rng(graine)
    return Graphe(
        pre=generateur.permutation(graphe.pre),
        post=graphe.post,
        synapses=graphe.synapses,
        signes=graphe.signes,
        n_neurones=graphe.n_neurones,
        entrees=graphe.entrees,
        sorties=graphe.sorties,
    )


def recabler_a_signes_conserves(graphe: Graphe, graine: int = 0) -> Graphe:
    """Permute à l'intérieur de chaque classe de signe (§11.2, contrôle facultatif).

    Chaque connexion garde alors son signe, donc chaque neurone garde exactement
    son équilibre excitation / inhibition : seule l'identité de la source change.
    L'écart entre ce contrôle et la mouche isole l'effet de la **topologie
    seule**, là où le recâblage ordinaire mélange deux effets.

    À ne lancer que si la v0.1 montre un écart entre la mouche et la mouche
    recâblée — sinon la question ne se pose pas.
    """
    generateur = np.random.default_rng(graine)
    nouveau = graphe.pre.copy()
    signes_des_sources = graphe.signes[graphe.pre]
    for signe in (1, -1):
        positions = np.flatnonzero(signes_des_sources == signe)
        if len(positions) > 1:
            nouveau[positions] = generateur.permutation(graphe.pre[positions])
    return Graphe(
        pre=nouveau,
        post=graphe.post,
        synapses=graphe.synapses,
        signes=graphe.signes,
        n_neurones=graphe.n_neurones,
        entrees=graphe.entrees,
        sorties=graphe.sorties,
    )


def anomalies_du_recablage(graphe: Graphe) -> dict[str, int]:
    """Boucles sur soi et doublons, à rapporter avec les résultats."""
    paires = np.stack([graphe.pre, graphe.post], axis=1)
    uniques = np.unique(paires, axis=0)
    return {
        "boucles_sur_soi": int((graphe.pre == graphe.post).sum()),
        "doublons": int(len(paires) - len(uniques)),
    }


def couper_le_graphe(mouche: Mouche) -> Mouche:
    """Gain **effectif** à 0 sur toutes les connexions (§11.2).

    Le gain effectif, pas le paramètre brut : une softplus ne rend jamais 0, il
    n'existe donc aucune valeur du paramètre qui coupe le graphe. On multiplie
    la sortie de la softplus.

    Les neurones moteurs, disjoints des neurones d'entrée, ne reçoivent alors
    plus rien : leur état reste nul, tous les candidats ont la même note, et le
    départage des égalités choisit uniformément. Le témoin doit donc rendre
    exactement le score du joueur au hasard.
    """
    mouche.echelle_des_gains = 0.0
    return mouche


def aveugler_l_entree(mouche: Mouche) -> Mouche:
    """Entrée sensorielle à 0 (§11.2).

    Le graphe fonctionne toujours, mais il ne reçoit aucune information sur la
    grille. Même conséquence attendue : toutes les notes égales, donc le hasard.
    Ce témoin-là attrape ce que le précédent ne voit pas — un chemin qui ferait
    entrer la grille autrement que par les 4 114 neurones sensoriels.
    """
    mouche.echelle_d_entree = 0.0
    return mouche
