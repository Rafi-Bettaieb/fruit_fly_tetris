"""La grille 10 × 20 : pose, disparition des lignes, mesures (conception.md §5.2, §6).

**Représentation.** Une grille est un tuple de 20 entiers, un par ligne, l'indice 0
en haut. Dans chaque entier, le bit `c` dit si la colonne `c` est occupée. C'est
immuable, donc chaque candidat produit une nouvelle grille sans jamais abîmer
celle d'origine — et comme on évalue jusqu'à 34 candidats sur la même grille de
départ, c'est exactement la propriété qu'il faut.

**Ce que `poser` rend.** La grille **après** la pose *et* après disparition des
lignes pleines, accompagnée du nombre de lignes disparues. Ce couple est
précisément ce qu'attend `encodage` (§9.1) : une fois les lignes parties, la
grille seule ne dit plus combien il y en avait, d'où les 5 entrées séparées.

**La chute est verticale, sans glissement** (§5.2). Une pièce ne peut donc jamais
se retrouver sous un surplomb : elle serait entrée en collision avec lui en
descendant. C'est ce qui permet de calculer le point d'arrivée directement, à
partir du sommet de chaque colonne, sans simuler la descente case par case.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .pieces import Rotation

LIGNES = 20
COLONNES = 10
LIGNE_PLEINE = (1 << COLONNES) - 1

Grille = tuple[int, ...]

GRILLE_VIDE: Grille = (0,) * LIGNES

# Valeur du « sommet » d'une colonne vide : aucune case occupée, donc le fond.
COLONNE_VIDE = LIGNES


def occupee(grille: Grille, ligne: int, colonne: int) -> bool:
    return bool((grille[ligne] >> colonne) & 1)


def sommets(grille: Grille) -> tuple[int, ...]:
    """Pour chaque colonne, l'indice de sa case occupée la plus haute.

    `COLONNE_VIDE` (= 20) si la colonne est vide. C'est sur ces valeurs que
    repose le calcul du point d'arrivée d'une pièce.
    """
    trouves = [COLONNE_VIDE] * COLONNES
    restantes = COLONNES
    for indice, ligne in enumerate(grille):
        if not ligne:
            continue
        for colonne in range(COLONNES):
            if trouves[colonne] == COLONNE_VIDE and (ligne >> colonne) & 1:
                trouves[colonne] = indice
                restantes -= 1
        if restantes == 0:
            break
    return tuple(trouves)


def hauteurs(grille: Grille) -> tuple[int, ...]:
    """La hauteur de chaque colonne, en cases. 0 pour une colonne vide."""
    return tuple(LIGNES - sommet for sommet in sommets(grille))


def hauteur_totale(grille: Grille) -> int:
    """Premier critère de l'expert (§10.1)."""
    return sum(hauteurs(grille))


def trous(grille: Grille) -> int:
    """Cases vides situées sous une case occupée, dans la même colonne.

    Troisième critère de l'expert, et la mesure qui explique pourquoi DAgger est
    indispensable : un trou en entraîne d'autres, et la mouche se retrouve vite
    dans des grilles que l'expert ne rencontre jamais (§10.4).
    """
    total = 0
    for colonne, sommet in enumerate(sommets(grille)):
        for ligne in range(sommet + 1, LIGNES):
            if not (grille[ligne] >> colonne) & 1:
                total += 1
    return total


def bosses(grille: Grille) -> int:
    """Somme des différences de hauteur entre colonnes voisines (§10.1)."""
    colonnes = hauteurs(grille)
    return sum(abs(colonnes[i] - colonnes[i + 1]) for i in range(COLONNES - 1))


class Mesures(NamedTuple):
    """Les trois mesures de grille dont l'expert a besoin (§10.1)."""

    hauteur_totale: int
    trous: int
    bosses: int


def mesures(grille: Grille) -> Mesures:
    """Calcule les trois d'un coup, en ne parcourant les sommets qu'une fois.

    L'expert les demande pour chacun des candidats d'une pièce ; les obtenir par
    trois appels séparés triplerait le parcours de la grille.
    """
    plafonds = sommets(grille)
    hauteur = 0
    vides = 0
    for colonne, sommet in enumerate(plafonds):
        hauteur += LIGNES - sommet
        for ligne in range(sommet + 1, LIGNES):
            if not (grille[ligne] >> colonne) & 1:
                vides += 1
    denivele = sum(abs(plafonds[i] - plafonds[i + 1]) for i in range(COLONNES - 1))
    return Mesures(hauteur_totale=hauteur, trous=vides, bosses=denivele)


def poser(
    grille: Grille,
    rotation: Rotation,
    colonne: int,
    plafonds: tuple[int, ...] | None = None,
) -> tuple[Grille, int] | None:
    """Fait tomber une rotation depuis le haut, à partir de `colonne`.

    Rend la grille d'après, avec le nombre de lignes disparues, ou `None` si le
    placement est impossible : soit la pièce déborde à droite, soit la pile est
    trop haute pour qu'elle tienne dans les 20 lignes. Ce second cas est ce qui
    met fin à la partie quand il vaut pour tous les placements (§5.2).

    `plafonds` évite de recalculer les sommets pour chacun des 34 candidats
    d'une même pièce : ils partent tous de la même grille. `candidats.enumerer`
    les calcule une fois et les passe ici.
    """
    if colonne < 0 or colonne + rotation.largeur > COLONNES:
        return None
    if plafonds is None:
        plafonds = sommets(grille)

    depart = LIGNES
    for colonne_piece, ligne_basse in rotation.colonnes_de_la_case_la_plus_basse().items():
        libre = plafonds[colonne + colonne_piece] - 1 - ligne_basse
        depart = min(depart, libre)

    if depart < 0:
        return None

    lignes = list(grille)
    for ligne_piece, colonne_piece in rotation.cases:
        lignes[depart + ligne_piece] |= 1 << (colonne + colonne_piece)

    # Seules les lignes que la pièce vient de toucher peuvent être devenues
    # pleines : au plus quatre, au lieu des vingt de la grille.
    pleines = {
        depart + decalage
        for decalage in range(rotation.hauteur)
        if lignes[depart + decalage] == LIGNE_PLEINE
    }
    if not pleines:
        return tuple(lignes), 0

    restantes = [ligne for indice, ligne in enumerate(lignes) if indice not in pleines]
    return (0,) * len(pleines) + tuple(restantes), len(pleines)


_MASQUES_DE_COLONNE = (1 << np.arange(COLONNES, dtype=np.int32))[None, :]


def cases_occupees(grille: Grille) -> np.ndarray:
    """Déplie la grille en un tableau booléen (20, 10), pour `encodage`.

    Dépliée en NumPy plutôt qu'avec une double boucle Python : cette fonction
    est appelée une fois par candidat, donc des centaines de millions de fois
    sur un entraînement complet.
    """
    return (np.array(grille, dtype=np.int32)[:, None] & _MASQUES_DE_COLONNE) != 0


def depuis_lignes(motifs: list[str]) -> Grille:
    """Construit une grille depuis un dessin, pour les tests et la mise au point.

    Chaque chaîne fait 10 caractères, `#` pour occupé, `.` pour vide, de haut en
    bas. Les lignes manquantes sont ajoutées vides **au-dessus**, de sorte qu'un
    dessin court décrive le bas de la grille.
    """
    if len(motifs) > LIGNES:
        raise ValueError(f"{len(motifs)} lignes pour une grille qui en compte {LIGNES}")
    lignes = []
    for motif in motifs:
        if len(motif) != COLONNES:
            raise ValueError(f"« {motif} » ne fait pas {COLONNES} caractères")
        lignes.append(sum(1 << c for c, signe in enumerate(motif) if signe == "#"))
    return (0,) * (LIGNES - len(lignes)) + tuple(lignes)
