"""La frontière entre le moteur et les modèles : 205 entrées (conception.md §9.1).

| Bloc              | Contenu                                  | Codage              | Taille |
|-------------------|------------------------------------------|---------------------|--------|
| Grille résultante | Les 200 cases après pose et disparition  | occupée +1, vide −1 | 200    |
| Lignes complétées | 0 à 4                                    | +1, les autres −1   | 5      |

Rien d'autre : ni la pièce suivante, ni le marquage de la pièce posée (§20).

Les 5 dernières entrées ne sont pas un supplément : une fois les lignes
disparues, la grille seule ne dit plus combien il y en avait. Sans elles, le
modèle ne peut pas distinguer un placement qui complète quatre lignes d'un
placement qui n'en complète aucune.

**Ce module est la seule chose que le moteur et les modèles partagent.** Le
moteur n'importe jamais torch ; les modèles ne connaissent jamais la grille.
C'est ce qui permet aux tests du moteur de tourner sans PyTorch, sans CUDA et
sans le MaleCNS (§16).
"""

from __future__ import annotations

import numpy as np

LIGNES = 20
COLONNES = 10
CASES = LIGNES * COLONNES  # 200
LIGNES_COMPLETEES_MAX = 4
TAILLE_BLOC_LIGNES = LIGNES_COMPLETEES_MAX + 1  # 5, pour 0..4
TAILLE_ENCODAGE = CASES + TAILLE_BLOC_LIGNES  # 205

OCCUPEE = 1.0
VIDE = -1.0


def encoder(grille_resultante, lignes_completees: int) -> np.ndarray:
    """Encode un candidat en un vecteur de 205 réels.

    `grille_resultante` est la grille **après** la pose et **après** disparition
    des lignes pleines. `lignes_completees` est le nombre de lignes que ce
    placement a fait disparaître, entre 0 et 4.
    """
    from .tetris.grille import cases_occupees

    vecteur = np.empty(TAILLE_ENCODAGE, dtype=np.float32)
    vecteur[:CASES] = np.where(cases_occupees(grille_resultante).ravel(), OCCUPEE, VIDE)
    vecteur[CASES:] = VIDE
    vecteur[CASES + lignes_completees] = OCCUPEE
    return vecteur


def encoder_lot(candidats) -> np.ndarray:
    """Encode tous les candidats d'une pièce, forme (n_candidats, 205).

    **C'est le seul chemin autorisé** entre le moteur et un modèle. Un noteur
    fondé sur un modèle ne lit rien d'autre du candidat : ni sa rotation, ni sa
    colonne, ni la pièce dont il vient. Sans cette règle, un modèle pourrait
    s'appuyer sur une régularité de l'énumération plutôt que sur la grille, et
    le contrôle « entrée aveuglée » ne le verrait pas (§9.3).
    """
    if not candidats:
        return np.zeros((0, TAILLE_ENCODAGE), dtype=np.float32)
    return np.stack(
        [encoder(candidat.grille, candidat.lignes_completees) for candidat in candidats]
    )
