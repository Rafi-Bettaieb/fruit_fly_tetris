"""Le nuage de neurones de la démo (§7.5, §13.2), sur les graphes construits en mémoire.

Ce qui est vérifié ici ne se voit pas à l'œil sur la page : que la vague parte
bien des neurones d'entrée, que l'image montre exactement l'état qui a servi à
noter, que les neurones sans soma soient placés là où arrivent leurs axones, et
que la page et le serveur parlent des mêmes neurones.
"""

from __future__ import annotations

import base64
import struct

import numpy as np
import pytest
import torch

from mouche_tetris import graines
from mouche_tetris.connectome import nuage as nu
from mouche_tetris.connectome.graphe import graphe_jouet, petit_connectome
from mouche_tetris.connectome.modele import Mouche
from mouche_tetris.encodage import encoder_lot
from mouche_tetris.enregistrement.capture import capturer
from mouche_tetris.tetris import candidats as c
from mouche_tetris.tetris import grille as g
from mouche_tetris.tetris.partie import jouer


@pytest.fixture(scope="module")
def reduit():
    return petit_connectome()


@pytest.fixture(scope="module")
def mouche(reduit):
    torch.manual_seed(0)
    return Mouche(reduit, k=6, facteur_global=4.0, graine_run=0)


def _encodages(n: int = 3) -> torch.Tensor:
    return torch.from_numpy(encoder_lot(tuple(c.enumerer(g.GRILLE_VIDE, "T")[:n])))


def _positions_factices(graphe, graine: int = 0) -> np.ndarray:
    """Un soma pour chaque neurone, sauf les neurones d'entrée — comme dans le MaleCNS."""
    generateur = graines.generateur_analyse(graine)
    somas = [list(generateur.integers(0, 10_000, size=3)) for _ in range(graphe.n_neurones)]
    for indice in graphe.entrees:
        somas[indice] = None
    return nu.positions_des_neurones(graphe, somas)


# --- La trajectoire du modèle -----------------------------------------------


def test_la_derniere_mise_a_jour_est_l_etat_qui_a_servi_a_noter(mouche, reduit):
    """L'image doit montrer exactement ce qui a été calculé pour décider."""
    indices = torch.arange(reduit.n_neurones)
    trajet = mouche.trajectoire(_encodages(), indices)
    assert trajet.shape == (3, mouche.k, reduit.n_neurones)
    with torch.no_grad():
        assert torch.allclose(trajet[:, -1], mouche.etats(_encodages()), atol=1e-6)


def test_la_vague_part_des_neurones_d_entree(mouche, reduit):
    """Après la première mise à jour, seuls les neurones sensoriels ont bougé :
    tous les états partent de zéro, et rien d'autre ne reçoit la grille."""
    premiere = mouche.trajectoire(_encodages(), torch.arange(reduit.n_neurones))[:, 0]
    actifs = set(np.flatnonzero((premiere.abs() > 0).any(dim=0).numpy()).tolist())
    assert actifs and actifs <= set(reduit.entrees.tolist())


def test_la_vague_atteint_les_neurones_moteurs(mouche, reduit):
    sorties = torch.from_numpy(reduit.sorties)
    trajet = mouche.trajectoire(_encodages(), sorties)
    assert float(trajet[:, 0].abs().max()) == 0.0
    assert float(trajet[:, -1].abs().max()) > 0.0


def test_la_note_n_a_pas_change(mouche):
    """Le refactoring de la boucle ne doit rien changer à la note."""
    with torch.no_grad():
        notes = mouche(_encodages(5))
        attendues = (mouche.etats(_encodages(5))[:, mouche.sorties]
                     * mouche.coefficients).sum(dim=1) * mouche.temperature
    assert torch.allclose(notes, attendues)


# --- Positions --------------------------------------------------------------


def test_un_neurone_d_entree_est_place_au_centre_de_ses_cibles():
    jouet = graphe_jouet()
    positions = _positions_factices(jouet)
    entree = int(jouet.entrees[0])
    aretes = jouet.pre == entree
    poids = jouet.synapses[aretes].astype(float)
    attendu = (positions[jouet.post[aretes]] * poids[:, None]).sum(axis=0) / poids.sum()
    if np.isnan(attendu).any():
        pytest.skip("cibles sans position dans ce tirage")
    assert np.allclose(positions[entree], attendu)


def test_les_positions_sont_en_micrometres():
    jouet = graphe_jouet()
    somas = [[1000, 2000, 3000]] * jouet.n_neurones
    positions = nu.positions_des_neurones(jouet, somas)
    assert np.allclose(positions[0], [8.0, 16.0, 24.0])


# --- Le tirage des neurones affichés ----------------------------------------


def test_le_tirage_garde_toutes_les_sorties_placees(reduit):
    positions = _positions_factices(reduit)
    nuage = nu.construire(reduit, positions, graine_run=0, taille=300, entrees=50)
    assert nuage.n == 300
    assert set(reduit.sorties.tolist()) <= set(nuage.indices.tolist())
    assert nuage.composition()["entrees"] == 50
    assert (nuage.roles[np.isin(nuage.indices, reduit.sorties)] == nu.SORTIE).all()
    assert (nuage.roles[np.isin(nuage.indices, reduit.entrees)] == nu.ENTREE).all()
    assert not np.isnan(nuage.positions).any()


def test_le_tirage_depend_de_la_graine_et_d_elle_seule(reduit):
    positions = _positions_factices(reduit)
    a = nu.construire(reduit, positions, graine_run=0, taille=300, entrees=50)
    b = nu.construire(reduit, positions, graine_run=0, taille=300, entrees=50)
    autre = nu.construire(reduit, positions, graine_run=1, taille=300, entrees=50)
    assert a.empreinte() == b.empreinte()
    assert a.empreinte() != autre.empreinte()


def test_le_flux_de_visualisation_n_est_pas_celui_du_run():
    """Tiré sur `generateur_run`, le nuage rejouerait les premiers tirages des
    interfaces et serait corrélé à la projection d'entrée."""
    assert (graines.generateur_visualisation(0).integers(0, 2**31, 8).tolist()
            != graines.generateur_run(0).integers(0, 2**31, 8).tolist())


def test_aller_retour_par_le_disque(reduit, tmp_path):
    nuage = nu.construire(reduit, _positions_factices(reduit), 0, taille=200, entrees=30)
    relu = nu.charger(nu.sauvegarder(nuage, tmp_path / "nuage.npz"))
    assert relu.empreinte() == nuage.empreinte()
    assert np.array_equal(relu.roles, nuage.roles)
    assert np.allclose(relu.positions, nuage.positions)


def test_le_format_de_la_page(reduit):
    nuage = nu.construire(reduit, _positions_factices(reduit), 0, taille=200, entrees=30)
    blob = nu.pour_la_page(nuage)
    assert blob[:6] == b"NUAGE1"
    (n,) = struct.unpack("<I", blob[6:10])
    assert n == nuage.n
    assert blob[10:22].decode("ascii") == nuage.empreinte()
    assert len(blob) == 6 + 4 + 12 + 16 + n * 6 + n
    reduites = np.frombuffer(blob[38:38 + n * 6], dtype="<i2").reshape(n, 3)
    assert np.abs(reduites).max() == 32767, "la plus grande demi-étendue remplit l'échelle"


# --- L'activité -------------------------------------------------------------


def test_quantifier():
    octets = nu.quantifier(np.array([0.0, 1.0, -1.0, 3.0, -3.0]))
    assert octets.tolist() == [128, 255, 1, 255, 0]


def test_l_activite_rapporte_chaque_neurone_a_sa_plage(mouche, reduit):
    nuage = nu.construire(reduit, _positions_factices(reduit), 0, taille=300, entrees=50)
    calibration = c.enumerer(g.GRILLE_VIDE, "L")
    activite = nu.Activite(mouche, nuage, calibration)
    etapes = activite.etapes(calibration[0])
    assert etapes.shape == (mouche.k, nuage.n) and etapes.dtype == np.uint8
    # Un neurone rapporté à sa propre plage atteint le bord de l'échelle sur au
    # moins une grille de calibration : sans cela, la normalisation serait globale.
    maxima = np.max([np.abs(activite.etapes(x).astype(int) - 128).max(axis=0)
                     for x in calibration], axis=0)
    actifs = activite.echelles.numpy() != 1.0
    assert actifs.any()
    assert (maxima[actifs] >= 126).all()


def test_le_noteur_avec_nuage_note_comme_les_autres(mouche, reduit):
    from mouche_tetris.modeles.torche import NoteurTorch

    nuage = nu.construire(reduit, _positions_factices(reduit), 0, taille=200, entrees=30)
    candidats = c.enumerer(g.GRILLE_VIDE, "T")
    avec = nu.NoteurAvecNuage(mouche, "mouche", "cpu", nu.Activite(mouche, nuage, candidats))
    assert np.allclose(avec.noter(candidats), NoteurTorch(mouche, "mouche").noter(candidats))
    activite = avec.activite(candidats[0])
    assert (activite["k"], activite["n"]) == (mouche.k, nuage.n)
    assert len(base64.b64decode(activite["etapes"])) == mouche.k * nuage.n
    assert avec.description_du_nuage()["empreinte"] == nuage.empreinte()


def test_une_partie_enregistree_porte_l_activite_de_chaque_coup(mouche, reduit):
    nuage = nu.construire(reduit, _positions_factices(reduit), 0, taille=200, entrees=30)
    avec = nu.NoteurAvecNuage(mouche, "mouche", "cpu",
                              nu.Activite(mouche, nuage, c.enumerer(g.GRILLE_VIDE, "I")))
    partie = jouer(avec, graine=1000, plafond=4, enregistrer_coups=True)
    capture = capturer(partie, "mouche", avec)
    assert capture["nuage"]["empreinte"] == nuage.empreinte()
    assert all(coup["neurones"]["k"] == mouche.k for coup in capture["coups"])


def test_sans_etat_interne_pas_de_nuage():
    """L'expert n'a pas de neurones : rien d'inventé à leur place."""
    from mouche_tetris.expert.criteres import Expert

    partie = jouer(Expert(), graine=1000, plafond=3, enregistrer_coups=True)
    capture = capturer(partie, "expert", Expert())
    assert capture["nuage"] is None
    assert all(coup["neurones"] is None for coup in capture["coups"])
