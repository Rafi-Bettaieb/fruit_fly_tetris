"""Construction du graphe depuis les fichiers bruts (conception.md §7.1).

Le fichier de poids contient les forces de connexion entre **tous** les
segments, fragments compris : environ 152 millions de lignes pour 25,6 millions
de connexions utiles. On ne garde que celles dont les deux extrémités sont des
neurones tracés.

**C'est le seul moment du projet où la RAM est en jeu** (§4.2). Chargé d'un
coup, le fichier et ses colonnes dérivées peuvent saturer les 24 Go de la
machine. On le lit donc par lots et l'on filtre **avant** de matérialiser quoi
que ce soit : seules les lignes retenues sont conservées, et la table complète
n'existe jamais en mémoire sous forme NumPy.

Le résultat est vérifié contre le contrat de §7.1 — 165 122 neurones,
25 563 197 connexions, 4 114 entrées, 708 sorties. Tout écart arrête le projet
plutôt que de le laisser continuer sur un graphe qui n'est pas celui de la
référence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .acquisition import BRUT, chemin, lire_les_neurones, verifier_le_graphe
from .graphe import Graphe
from .signes import lire_les_signes

TAILLE_DE_LOT = 4_000_000


def colonnes_du_fichier_de_poids(racine: Path = BRUT) -> list[str]:
    import pyarrow.feather as feather

    return list(feather.read_table(chemin("poids", racine), memory_map=True).column_names)


def _deviner_colonnes(colonnes: list[str]) -> tuple[str, str, str]:
    """Repère les colonnes pré, post et poids quel que soit leur nom exact.

    Les noms n'étant pas garantis d'une version à l'autre, on les cherche par
    motif plutôt que de les coder en dur — et l'on échoue clairement si l'on
    ne trouve pas, au lieu de deviner de travers.
    """
    minuscules = {nom.lower(): nom for nom in colonnes}

    def trouver(*motifs: str) -> str | None:
        for motif in motifs:
            for bas, original in minuscules.items():
                if motif in bas:
                    return original
        return None

    pre = trouver("bodyid_pre", "body_pre", "pre_id", "pre")
    post = trouver("bodyid_post", "body_post", "post_id", "post")
    poids = trouver("weight", "synapses", "count", "strength")
    if not (pre and post and poids):
        raise RuntimeError(
            f"colonnes introuvables dans le fichier de poids : {colonnes}\n"
            "Renseigner les noms à la main dans `construire_le_graphe`."
        )
    return pre, post, poids


def construire_le_graphe(racine: Path = BRUT, verifier: bool = True) -> Graphe:
    """Lit les trois fichiers et rend le graphe prêt pour le modèle."""
    import pyarrow.feather as feather

    neurones = lire_les_neurones(racine)
    identifiants = neurones["identifiants"]

    # Correspondance identifiant MaleCNS → indice dense 0..165 121. Les `bodyId`
    # montent à 1 471 062 202 : une table dense pèserait 1,5 Go pour 165 122
    # entrées utiles. On trie une fois et l'on cherche par dichotomie — 1,3 Mo.
    ordre = np.argsort(identifiants)
    tries = identifiants[ordre]
    rang_vers_indice = ordre

    def vers_indice_dense(bruts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rend (masque des identifiants tracés, indices denses correspondants)."""
        position = np.searchsorted(tries, bruts)
        np.clip(position, 0, len(tries) - 1, out=position)
        connu = tries[position] == bruts
        return connu, rang_vers_indice[position]

    table = feather.read_table(chemin("poids", racine), memory_map=True)
    colonne_pre, colonne_post, colonne_poids = _deviner_colonnes(list(table.column_names))

    morceaux_pre: list[np.ndarray] = []
    morceaux_post: list[np.ndarray] = []
    morceaux_poids: list[np.ndarray] = []

    for lot in table.to_batches(max_chunksize=TAILLE_DE_LOT):
        pre = lot.column(colonne_pre).to_numpy(zero_copy_only=False).astype(np.int64)
        post = lot.column(colonne_post).to_numpy(zero_copy_only=False).astype(np.int64)
        poids = lot.column(colonne_poids).to_numpy(zero_copy_only=False).astype(np.float64)

        # Filtrage avant toute matérialisation : c'est ce qui tient dans la RAM.
        pre_connu, pre_indices = vers_indice_dense(pre)
        post_connu, post_indices = vers_indice_dense(post)
        garde = pre_connu & post_connu
        if not garde.any():
            continue
        morceaux_pre.append(pre_indices[garde])
        morceaux_post.append(post_indices[garde])
        morceaux_poids.append(poids[garde])

    graphe = Graphe(
        pre=np.concatenate(morceaux_pre),
        post=np.concatenate(morceaux_post),
        synapses=np.concatenate(morceaux_poids),
        signes=lire_les_signes(identifiants, racine),
        n_neurones=len(identifiants),
        entrees=vers_indice_dense(neurones["entrees"])[1],
        sorties=vers_indice_dense(neurones["sorties"])[1],
    )

    if verifier:
        verifier_le_graphe(
            graphe.n_neurones, graphe.n_connexions, len(graphe.entrees), len(graphe.sorties)
        )
    return graphe


PREPARE_GRAPHE = Path("data/prepare/graphe.pkl")
"""Le graphe préparé, hors git. Quelques centaines de Mo, contre 1,2 Go de brut."""
