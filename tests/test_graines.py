"""Les graines : le premier test du projet, parce que c'est le premier piège.

Une collision entre deux plages voudrait dire qu'on évalue sur des parties vues
à l'entraînement. Un générateur partagé entre les pièces et le départage
voudrait dire que deux conditions ne jouent plus les mêmes séquences.

Ni l'une ni l'autre ne lève d'erreur à l'exécution. D'où ces tests.
"""

from __future__ import annotations

from mouche_tetris import graines


def test_plages_de_graines_disjointes():
    graines.verifier_plages_disjointes()


def test_la_sequence_de_pieces_ne_depend_que_de_la_partie():
    """Deux joueurs sur la graine 1000 doivent voir exactement la même séquence."""
    a = graines.generateur_pieces(1000).integers(0, 7, size=500)
    b = graines.generateur_pieces(1000).integers(0, 7, size=500)
    assert (a == b).all()


def test_pieces_et_departage_sont_des_flux_distincts():
    """Tirer au sort entre deux placements ne doit jamais décaler les pièces."""
    pieces = graines.generateur_pieces(1000).integers(0, 7, size=500)
    departage = graines.generateur_departage(1000).integers(0, 7, size=500)
    assert not (pieces == departage).all()


def test_deux_parties_differentes_donnent_des_sequences_differentes():
    a = graines.generateur_pieces(1000).integers(0, 7, size=500)
    b = graines.generateur_pieces(1001).integers(0, 7, size=500)
    assert not (a == b).all()
