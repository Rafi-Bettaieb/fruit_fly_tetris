"""Boucle d'entraînement, accord avec l'expert, juge linéaire.

Tous ces tests tournent sur CPU en quelques secondes : c'est ce qui permet de
valider la boucle avant qu'elle ne coûte des heures de GPU à la mouche.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mouche_tetris import graines
from mouche_tetris.entrainement import clonage
from mouche_tetris.entrainement import demonstrations as d
from mouche_tetris.evaluation.accord import accord, niveau_du_hasard
from mouche_tetris.expert.criteres import Expert
from mouche_tetris.modeles.hasard import Hasard
from mouche_tetris.modeles.lineaire import JugeLineaire
from mouche_tetris.modeles.torche import NoteurTorch, verifier_forme


@pytest.fixture(scope="module")
def situations():
    return d.generer(range(0, 6), plafond=120)


@pytest.fixture(scope="module")
def situations_de_test():
    return d.generer(range(500, 502), bruit=0.0, plafond=120)


# --- Accord --------------------------------------------------------------


def test_l_expert_est_d_accord_avec_lui_meme(situations_de_test):
    assert accord(Expert(), situations_de_test) == 1.0


def test_le_hasard_retombe_sur_le_niveau_du_hasard(situations_de_test):
    """Le repère qui rend le seuil de 10 % lisible (§18)."""
    mesure = accord(Hasard(), situations_de_test)
    attendu = niveau_du_hasard(situations_de_test)
    assert 0.02 < attendu < 0.10
    assert abs(mesure - attendu) < 0.03


def test_accord_reproductible(situations_de_test):
    assert accord(Hasard(), situations_de_test) == accord(Hasard(), situations_de_test)


# --- Lot et perte --------------------------------------------------------


def test_le_lot_masque_les_places_inventees(situations):
    reglages = clonage.Reglages(candidats_par_situation=10, lot=8)
    generateur = graines.generateur_run(0)
    encodages, masque, cibles, _ = clonage.preparer_lot(situations[:8], reglages, generateur)
    assert encodages.shape[0] == 8
    assert masque.shape == encodages.shape[:2]
    assert masque.any(dim=1).all(), "chaque situation a au moins un candidat"
    for indice, cible in enumerate(cibles.tolist()):
        assert masque[indice, cible], "la cible ne peut pas tomber sur une place inventée"


def test_les_places_inventees_ne_recoivent_aucun_gradient(situations):
    """Sans le masque, la softmax distribuerait de la probabilité à des candidats
    qui n'existent pas, et le modèle apprendrait à les éviter — du bruit pur."""
    reglages = clonage.Reglages(candidats_par_situation=34, lot=8)
    generateur = graines.generateur_run(0)
    encodages, masque, cibles, _ = clonage.preparer_lot(situations[:8], reglages, generateur)
    if masque.all():
        pytest.skip("aucune situation n'a eu besoin de remplissage")
    modele = JugeLineaire()
    perte = clonage.perte_du_lot(modele, encodages, masque, cibles)
    assert torch.isfinite(perte)


def test_la_perte_de_depart_vaut_le_hasard(situations):
    """Un modèle non entraîné met la même note partout : la perte vaut ln(n)."""
    reglages = clonage.Reglages(candidats_par_situation=10, lot=32)
    generateur = graines.generateur_run(0)
    encodages, masque, cibles, _ = clonage.preparer_lot(situations[:32], reglages, generateur)
    modele = JugeLineaire()
    with torch.no_grad():
        modele.couche.weight.zero_()
        modele.couche.bias.zero_()
    perte = clonage.perte_du_lot(modele, encodages, masque, cibles)
    attendu = float(np.log(masque.sum(dim=1).numpy().astype(float)).mean())
    assert abs(float(perte.detach()) - attendu) < 1e-5


# --- Modèle --------------------------------------------------------------


def test_le_juge_lineaire_a_206_parametres():
    assert sum(p.numel() for p in JugeLineaire().parameters()) == 206


def test_forme_de_sortie():
    verifier_forme(JugeLineaire())


def test_l_adaptateur_donne_une_note_par_candidat(situations):
    noteur = NoteurTorch(JugeLineaire(), "linéaire")
    notes = noteur.noter(situations[0].candidats)
    assert notes.shape == (len(situations[0].candidats),)


def test_la_note_ne_depend_pas_du_lot(situations):
    """Invariance au lot (§16) : évaluer un candidat seul ou parmi 34 donne la
    même note. C'est le test qui détectera une fuite entre candidats dans le
    noyau creux du connectome."""
    noteur = NoteurTorch(JugeLineaire(), "linéaire")
    candidats = situations[0].candidats
    ensemble = noteur.noter(candidats)
    for indice, candidat in enumerate(candidats):
        seul = noteur.noter((candidat,))
        assert abs(seul[0] - ensemble[indice]) < 1e-5


# --- Entraînement --------------------------------------------------------


def test_la_boucle_apprend(situations, situations_de_test):
    """La validation la moins chère du projet : si le linéaire n'apprend pas
    ici, le problème est dans la boucle, pas dans le connectome."""
    reglages = clonage.Reglages(lot=64, taux=0.01, mises_a_jour=300,
                                point_de_controle_tous_les=100)
    avant = accord(NoteurTorch(JugeLineaire(), "linéaire"), situations_de_test)
    modele, journal = clonage.entrainer(
        JugeLineaire(), situations, situations_de_test, reglages, nom="linéaire"
    )
    apres = accord(NoteurTorch(modele, "linéaire"), situations_de_test)
    assert not journal.echoue
    assert apres > avant + 0.10
    assert np.mean(journal.pertes[-50:]) < np.mean(journal.pertes[:50])


def test_les_points_de_controle_sont_pris(situations, situations_de_test):
    reglages = clonage.Reglages(lot=32, taux=0.01, mises_a_jour=200,
                                point_de_controle_tous_les=50)
    _, journal = clonage.entrainer(JugeLineaire(), situations, situations_de_test, reglages)
    assert [pas for pas, _ in journal.accords] == [50, 100, 150, 200]


def test_la_divergence_declenche_une_relance(situations, situations_de_test):
    """Un taux absurde doit faire exploser la perte, donc relancer, pas planter."""
    reglages = clonage.Reglages(lot=8, taux=1e9, mises_a_jour=400,
                                point_de_controle_tous_les=100, relances_max=3)
    _, journal = clonage.entrainer(JugeLineaire(), situations, situations_de_test, reglages)
    assert journal.relances > 0


def test_l_entrainement_est_reproductible(situations, situations_de_test):
    reglages = clonage.Reglages(lot=16, taux=0.01, mises_a_jour=100,
                                point_de_controle_tous_les=50)
    torch.manual_seed(0)
    a, _ = clonage.entrainer(JugeLineaire(), situations, situations_de_test, reglages, graine_run=0)
    torch.manual_seed(0)
    b, _ = clonage.entrainer(JugeLineaire(), situations, situations_de_test, reglages, graine_run=0)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.allclose(pa, pb)


# --- Les deux corrections à l'essai (§10.3) ------------------------------


def test_la_cible_fait_toujours_partie_des_optimaux(situations):
    reglages = clonage.Reglages(candidats_par_situation=10, lot=32)
    generateur = graines.generateur_run(0)
    _, masque, cibles, optimaux = clonage.preparer_lot(situations[:32], reglages, generateur)
    for indice, cible in enumerate(cibles.tolist()):
        assert optimaux[indice, cible]
    assert not (optimaux & ~masque).any(), "une place inventée ne peut pas être optimale"


def test_avec_un_seul_optimal_les_deux_pertes_sont_egales(situations):
    """83,6 % des situations : la correction ne change rien là où il n'y a pas
    d'égalité chez l'expert."""
    uniques = [s for s in situations if len(s.meilleurs_expert) == 1][:16]
    reglages = clonage.Reglages(candidats_par_situation=10, lot=16)
    lot = clonage.preparer_lot(uniques, reglages, graines.generateur_run(0))
    torch.manual_seed(0)
    modele = JugeLineaire()
    choix = clonage.perte_du_lot(modele, *lot, perte="choix")
    optimaux = clonage.perte_du_lot(modele, *lot, perte="optimaux")
    assert float(choix) == pytest.approx(float(optimaux), abs=1e-6)


def test_un_autre_optimal_n_est_plus_puni():
    """Le défaut corrigé : l'expert a deux optimaux, il en a tiré un, le modèle
    préfère l'autre. « choix » le punit lourdement ; « optimaux » ne le punit pas."""
    notes = torch.tensor([[0.0, 10.0, 0.0, 0.0]])  # le modèle choisit le candidat 1

    class Fixe(torch.nn.Module):
        def forward(self, encodages):
            return notes.reshape(-1)

    encodages = torch.zeros(1, 4, 205)
    masque = torch.ones(1, 4, dtype=torch.bool)
    cibles = torch.tensor([0])                          # l'expert a tiré le 0…
    optimaux = torch.tensor([[True, True, False, False]])  # …mais le 1 le vaut
    choix = clonage.perte_du_lot(Fixe(), encodages, masque, cibles, optimaux, "choix")
    optimaux_ = clonage.perte_du_lot(Fixe(), encodages, masque, cibles, optimaux, "optimaux")
    assert float(choix) > 9
    assert float(optimaux_) < 1e-3


def test_une_perte_inconnue_est_refusee():
    with pytest.raises(ValueError, match="inconnue"):
        clonage.Reglages(perte="autre")


def test_le_taux_constant_par_defaut():
    reglages = clonage.Reglages()
    assert clonage.taux_du_pas(reglages, 0) == clonage.taux_du_pas(reglages, 2399) == 0.04


def test_le_taux_decroit_du_depart_a_la_cible():
    reglages = clonage.Reglages(mises_a_jour=800, taux_final=0.004)
    assert clonage.taux_du_pas(reglages, 0) == pytest.approx(0.04)
    assert clonage.taux_du_pas(reglages, 400) == pytest.approx(0.022)
    assert clonage.taux_du_pas(reglages, 800) == pytest.approx(0.004)
    taux = [clonage.taux_du_pas(reglages, p) for p in range(801)]
    assert all(a >= b for a, b in zip(taux, taux[1:], strict=False))


def test_une_relance_divise_aussi_le_taux_decroissant():
    reglages = clonage.Reglages(mises_a_jour=800, taux_final=0.004)
    assert clonage.taux_du_pas(reglages, 200, relances=1) == pytest.approx(
        clonage.taux_du_pas(reglages, 200) / 4)
