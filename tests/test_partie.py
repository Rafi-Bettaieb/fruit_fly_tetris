"""Sacs de 7, boucle de jeu, départage des égalités.

Le test le plus important du fichier est `test_meme_sequence_quel_que_soit_le_joueur` :
c'est lui qui garantit l'évaluation appariée, et son échec ne se verrait nulle
part ailleurs.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from mouche_tetris import graines
from mouche_tetris.expert.criteres import Expert
from mouche_tetris.modeles.hasard import Hasard
from mouche_tetris.tetris import partie as p
from mouche_tetris.tetris import pieces, sacs

# --- Sacs de 7 -----------------------------------------------------------


def test_chaque_sac_contient_les_sept_pieces():
    suite = sacs.premieres(0, 70)
    for debut in range(0, 70, 7):
        assert set(suite[debut : debut + 7]) == set(pieces.LETTRES)


def test_frequences_egales_sur_un_nombre_entier_de_sacs():
    """Aucune condition ne peut recevoir plus de I qu'une autre."""
    compte = Counter(sacs.premieres(0, 700))
    assert set(compte.values()) == {100}


def test_meme_graine_meme_sequence():
    assert sacs.premieres(1000, 200) == sacs.premieres(1000, 200)


def test_graines_differentes_sequences_differentes():
    assert sacs.premieres(1000, 200) != sacs.premieres(1001, 200)


def test_meme_sequence_quel_que_soit_le_joueur():
    """Le cœur de l'évaluation appariée (§11.1).

    L'expert et le joueur au hasard jouent la graine 1500 : ils doivent voir
    exactement la même suite de pièces, alors que leurs parties n'ont rien à
    voir. Si le départage des égalités puisait dans le flux des pièces, ce test
    échouerait — et lui seul le verrait.
    """
    partie_expert = p.jouer(Expert(), graine=1500, plafond=60, enregistrer_coups=True)
    partie_hasard = p.jouer(Hasard(), graine=1500, plafond=60, enregistrer_coups=True)
    suite_expert = [coup.lettre for coup in partie_expert.coups]
    suite_hasard = [coup.lettre for coup in partie_hasard.coups]
    commun = min(len(suite_expert), len(suite_hasard))
    assert commun > 0
    assert suite_expert[:commun] == suite_hasard[:commun]
    assert suite_expert[:commun] == list(sacs.premieres(1500, commun))


# --- Départage -----------------------------------------------------------


def test_departage_sans_egalite():
    notes = np.array([1.0, 3.0, 2.0])
    generateur = graines.generateur_departage(0)
    assert p.choisir(notes, generateur) == 1


def test_departage_avec_egalite_reste_dans_les_meilleurs():
    notes = np.array([5.0, 5.0, 1.0, 5.0])
    generateur = graines.generateur_departage(0)
    for _ in range(50):
        assert p.choisir(notes, generateur) in {0, 1, 3}


def test_departage_uniforme_quand_tout_est_a_egalite():
    """C'est ce qui fait du joueur au hasard un vrai joueur au hasard."""
    notes = np.zeros(10)
    generateur = graines.generateur_departage(0)
    tires = Counter(p.choisir(notes, generateur) for _ in range(4000))
    assert set(tires) == set(range(10))
    assert min(tires.values()) > 250, "tirage trop déséquilibré"


# --- Boucle de jeu -------------------------------------------------------


def test_le_plafond_arrete_la_partie():
    partie = p.jouer(Expert(), graine=1500, plafond=40)
    assert partie.pieces_posees == 40
    assert partie.plafond_atteint


def test_le_hasard_perd_avant_le_plafond():
    partie = p.jouer(Hasard(), graine=1500, plafond=500)
    assert not partie.plafond_atteint
    assert partie.pieces_posees < 200


def test_une_partie_est_reproductible():
    a = p.jouer(Expert(), graine=1500, plafond=80)
    b = p.jouer(Expert(), graine=1500, plafond=80)
    assert (a.lignes, a.pieces_posees, a.trous_crees) == (b.lignes, b.pieces_posees, b.trous_crees)


def test_les_coups_ne_sont_gardes_que_sur_demande():
    assert p.jouer(Expert(), graine=1500, plafond=20).coups == ()
    assert len(p.jouer(Expert(), graine=1500, plafond=20, enregistrer_coups=True).coups) == 20


def test_un_noteur_qui_rend_le_mauvais_nombre_de_notes_est_refuse():
    class Cassé:
        nom = "cassé"

        def noter(self, candidats):
            return np.zeros(3)

    with pytest.raises(ValueError, match="notes"):
        p.jouer(Cassé(), graine=1500, plafond=5)


# --- L'expert joue mieux que le hasard -----------------------------------


def test_l_expert_ecrase_le_hasard():
    expert = p.jouer(Expert(), graine=1500, plafond=300)
    hasard = p.jouer(Hasard(), graine=1500, plafond=300)
    assert expert.lignes > hasard.lignes
    assert expert.pieces_posees > hasard.pieces_posees
    assert expert.trous_par_piece < hasard.trous_par_piece
