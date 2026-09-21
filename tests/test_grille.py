"""La grille : pose, disparition des lignes, mesures.

Les bugs visés ici sont tous silencieux. Une pièce qui se glisse sous un
surplomb, un décalage d'une ligne à la disparition, une grille rendue *avant*
nettoyage plutôt qu'après : rien de tout cela ne lève d'erreur, et tout fausse
l'encodage des 205 entrées, donc toutes les mesures qui suivent.
"""

from __future__ import annotations

import pytest

from mouche_tetris.tetris import grille as g
from mouche_tetris.tetris import pieces


def rotation(lettre: str, indice: int = 0):
    return pieces.rotations(lettre)[indice]


def i_vertical():
    return next(r for r in pieces.rotations("I") if r.largeur == 1)


def i_horizontal():
    return next(r for r in pieces.rotations("I") if r.largeur == 4)


# --- Grille vide ---------------------------------------------------------


def test_grille_vide():
    assert g.hauteurs(g.GRILLE_VIDE) == (0,) * 10
    assert g.hauteur_totale(g.GRILLE_VIDE) == 0
    assert g.trous(g.GRILLE_VIDE) == 0
    assert g.bosses(g.GRILLE_VIDE) == 0
    assert g.sommets(g.GRILLE_VIDE) == (g.COLONNE_VIDE,) * 10


def test_dessin_decrit_le_bas_de_la_grille():
    grille = g.depuis_lignes(["##........"])
    assert g.occupee(grille, g.LIGNES - 1, 0)
    assert g.occupee(grille, g.LIGNES - 1, 1)
    assert not g.occupee(grille, g.LIGNES - 2, 0)


# --- Chute ---------------------------------------------------------------


def test_une_piece_tombe_jusqu_au_fond():
    resultat = g.poser(g.GRILLE_VIDE, rotation("O"), 0)
    assert resultat is not None
    nouvelle, completees = resultat
    assert completees == 0
    assert g.hauteurs(nouvelle)[:2] == (2, 2)
    assert g.hauteurs(nouvelle)[2:] == (0,) * 8


def test_une_piece_se_pose_sur_la_pile():
    base = g.depuis_lignes(["##........"])
    nouvelle, _ = g.poser(base, rotation("O"), 0)
    assert g.hauteurs(nouvelle)[:2] == (3, 3)


def test_une_piece_ne_se_glisse_jamais_sous_un_surplomb():
    """La chute est verticale : la pièce s'arrête sur le surplomb, pas dessous.

    Colonne 5 occupée en haut, vide en dessous. Un O lâché là doit se poser
    au-dessus du surplomb, en laissant les trous intacts.
    """
    base = g.depuis_lignes(["." * 10] * 9 + [".....#...."] + ["." * 10] * 10)
    trous_avant = g.trous(base)
    nouvelle, completees = g.poser(base, rotation("O"), 5)
    assert completees == 0
    assert g.sommets(nouvelle)[5] == 7, "le O doit reposer sur le surplomb"
    assert g.trous(nouvelle) >= trous_avant, "aucun trou n'a été rebouché par en dessous"


def test_placement_qui_deborde_a_droite():
    """Une pièce de largeur `l` va jusqu'à la colonne `10 − l`, pas au-delà."""
    assert g.poser(g.GRILLE_VIDE, i_horizontal(), 6) is not None  # colonnes 6 à 9
    assert g.poser(g.GRILLE_VIDE, i_horizontal(), 7) is None  # déborderait en 10
    assert g.poser(g.GRILLE_VIDE, rotation("O"), 8) is not None  # colonnes 8 et 9
    assert g.poser(g.GRILLE_VIDE, rotation("O"), 9) is None


def test_placement_impossible_quand_la_pile_touche_le_haut():
    """C'est ce cas, valable pour tous les placements, qui met fin à la partie."""
    pleine_sauf_une = ["#########." for _ in range(g.LIGNES)]
    base = g.depuis_lignes(pleine_sauf_une)
    assert g.poser(base, i_vertical(), 0) is None


# --- Disparition des lignes ----------------------------------------------


def test_une_ligne_complete_disparait():
    base = g.depuis_lignes(["#########."])
    nouvelle, completees = g.poser(base, i_vertical(), 9)
    assert completees == 1
    # Il reste les trois cases du I au-dessus, dans la colonne 9.
    assert g.hauteurs(nouvelle) == (0,) * 9 + (3,)


def test_quatre_lignes_d_un_coup():
    """Le cas limite : un I vertical dans un puits de quatre lignes."""
    base = g.depuis_lignes(["#########."] * 4)
    nouvelle, completees = g.poser(base, i_vertical(), 9)
    assert completees == 4
    assert nouvelle == g.GRILLE_VIDE


def test_la_grille_rendue_est_celle_d_apres_la_disparition():
    """Contrat avec `encodage` : la grille est nettoyée, le compte est à part (§9.1)."""
    base = g.depuis_lignes(["########.."])
    nouvelle, completees = g.poser(base, rotation("O"), 8)
    assert completees == 1
    for ligne in nouvelle:
        assert ligne != g.LIGNE_PLEINE
    # Le haut du O survit à la disparition de la ligne du bas.
    assert g.hauteurs(nouvelle)[8:] == (1, 1)


def test_les_lignes_du_dessus_descendent():
    base = g.depuis_lignes(["#........."] + ["#########."])
    nouvelle, completees = g.poser(base, i_vertical(), 9)
    assert completees == 1
    # La case isolée de la colonne 0 est descendue d'une ligne, jusqu'au fond.
    assert g.hauteurs(nouvelle)[0] == 1


# --- Mesures de l'expert -------------------------------------------------


def test_trous():
    base = g.depuis_lignes(["#........#", ".........#"])
    assert g.trous(base) == 1  # colonne 0 : rien dessous ; colonne 9 : pleine


def test_trous_compte_chaque_case_vide_sous_un_bloc():
    base = g.depuis_lignes(["#........."] + [".........."] * 2)
    assert g.trous(base) == 2


def test_bosses():
    base = g.depuis_lignes(["#........."])
    # hauteurs : 1 puis neuf 0 → une seule différence de 1
    assert g.bosses(base) == 1
    assert g.hauteur_totale(base) == 1


# --- Frontière avec l'encodage -------------------------------------------


def test_cases_occupees_a_la_bonne_forme():
    tableau = g.cases_occupees(g.depuis_lignes(["#........."]))
    assert tableau.shape == (g.LIGNES, g.COLONNES)
    assert tableau.sum() == 1
    assert tableau[g.LIGNES - 1, 0]


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_toute_piece_se_pose_sur_grille_vide(lettre):
    """Sur une grille vide, tous les placements comptés en §6 doivent réussir."""
    valides = 0
    for rot in pieces.rotations(lettre):
        for colonne in range(g.COLONNES - rot.largeur + 1):
            resultat = g.poser(g.GRILLE_VIDE, rot, colonne)
            assert resultat is not None
            valides += 1
    assert valides == pieces.placements_sur_grille_vide(lettre)


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_une_pose_ajoute_toujours_quatre_cases(lettre):
    """Sauf disparition de ligne, une pièce ajoute exactement ses quatre cases."""
    for rot in pieces.rotations(lettre):
        nouvelle, completees = g.poser(g.GRILLE_VIDE, rot, 0)
        assert completees == 0
        assert g.cases_occupees(nouvelle).sum() == 4
