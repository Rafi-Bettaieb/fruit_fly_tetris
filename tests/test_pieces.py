"""La géométrie des pièces, vérifiée contre les faits de conception.md §6.

Ces tests ne dépendent d'aucun choix d'implémentation : 9, 17 et 34 se
démontrent à la main. Si l'un d'eux tombe, c'est le code qui a tort.
"""

from __future__ import annotations

import pytest

from mouche_tetris.tetris import pieces

# Les chiffres du document (§6), et leur justification en une ligne chacun.
PLACEMENTS_ATTENDUS = {
    "O": 9,  # 1 rotation, largeur 2            → 9
    "I": 17,  # largeurs 4 et 1                  → 7 + 10
    "S": 17,  # largeurs 3 et 2                  → 8 + 9
    "Z": 17,  # largeurs 3 et 2                  → 8 + 9
    "T": 34,  # largeurs 3, 2, 3, 2              → 8 + 9 + 8 + 9
    "J": 34,
    "L": 34,
}

ROTATIONS_ATTENDUES = {"O": 1, "I": 2, "S": 2, "Z": 2, "T": 4, "J": 4, "L": 4}


def test_sept_pieces():
    assert set(pieces.LETTRES) == set(PLACEMENTS_ATTENDUS)
    assert len(pieces.LETTRES) == 7


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_chaque_rotation_a_quatre_cases(lettre):
    """Un tétromino a quatre cases, dans toutes ses orientations."""
    for rotation in pieces.rotations(lettre):
        assert len(rotation.cases) == 4
        assert len(set(rotation.cases)) == 4, "deux cases superposées"


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_nombre_de_rotations_distinctes(lettre):
    """Le O n'a qu'une forme, le I, le S et le Z en ont deux, le reste quatre.

    Garder les doublons ferait se partager la masse de la softmax entre des
    candidats identiques pendant l'entraînement (§10.3).
    """
    assert len(pieces.rotations(lettre)) == ROTATIONS_ATTENDUES[lettre]


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_rotations_collees_en_haut_a_gauche(lettre):
    """Chaque rotation touche la ligne 0 et la colonne 0.

    Sans cette normalisation, deux rotations identiques mais décalées
    passeraient pour distinctes.
    """
    for rotation in pieces.rotations(lettre):
        assert min(ligne for ligne, _ in rotation.cases) == 0
        assert min(colonne for _, colonne in rotation.cases) == 0


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_placements_sur_grille_vide(lettre):
    """Le fait central de §6 : O 9 ; I, S, Z 17 ; T, J, L 34."""
    assert pieces.placements_sur_grille_vide(lettre) == PLACEMENTS_ATTENDUS[lettre]


def test_jamais_plus_de_34_candidats():
    """La borne sur laquelle reposent le lot d'évaluation et le budget mémoire.

    34 candidats en parallèle, c'est ce que dimensionnent §9.4 et §14.1.
    """
    assert max(pieces.placements_sur_grille_vide(lettre) for lettre in pieces.LETTRES) == 34


def test_moyenne_de_23_candidats():
    """La moyenne sur les sept pièces, utilisée dans les estimations de temps (§14.3)."""
    total = sum(pieces.placements_sur_grille_vide(lettre) for lettre in pieces.LETTRES)
    moyenne = total / 7
    assert moyenne == pytest.approx(23.1, abs=0.05)


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_case_la_plus_basse_par_colonne(lettre):
    """La chute verticale s'appuie dessus : une entrée par colonne occupée."""
    for rotation in pieces.rotations(lettre):
        plus_basses = rotation.colonnes_de_la_case_la_plus_basse()
        assert set(plus_basses) == {colonne for _, colonne in rotation.cases}
        assert len(plus_basses) == rotation.largeur
