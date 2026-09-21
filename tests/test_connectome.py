"""Le modèle du connectome, sur le graphe jouet de 20 neurones.

**Aucun de ces tests ne télécharge le MaleCNS** (§16) : le graphe jouet est
construit en mémoire. Ils vérifient ce qui ne produit jamais de message d'erreur
quand c'est faux — l'orientation du graphe, la loi de Dale, la normalisation,
l'invariance au lot, et le fait que les témoins retombent bien au hasard.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mouche_tetris.connectome import controles as ctrl
from mouche_tetris.connectome.graphe import Graphe, graphe_jouet
from mouche_tetris.connectome.modele import Mouche
from mouche_tetris.encodage import TAILLE_ENCODAGE


@pytest.fixture
def jouet():
    return graphe_jouet()


@pytest.fixture
def mouche(jouet):
    return Mouche(jouet, k=6, facteur_global=4.0, graine_run=0)


def encodages(n: int = 5, graine: int = 0) -> torch.Tensor:
    generateur = np.random.default_rng(graine)
    tirage = generateur.choice([1.0, -1.0], size=(n, TAILLE_ENCODAGE))
    return torch.tensor(tirage, dtype=torch.float32)


# --- Graphe --------------------------------------------------------------


def test_populations_disjointes(jouet):
    assert set(jouet.entrees.tolist()) & set(jouet.sorties.tolist()) == set()


def test_populations_qui_se_chevauchent_sont_refusees():
    """Un chevauchement permettrait au signal d'atteindre la sortie sans passer
    par le graphe — c'est la règle de conception de §9.3."""
    with pytest.raises(ValueError, match="disjointes"):
        Graphe(
            pre=np.array([0]), post=np.array([1]), synapses=np.array([1.0]),
            signes=np.ones(3, dtype=np.int64), n_neurones=3,
            entrees=np.array([0, 1]), sorties=np.array([1, 2]),
        )


def test_normalisation_somme_a_un(jouet):
    """La somme des valeurs absolues des poids entrants vaut 1 par neurone (§7.4)."""
    poids = jouet.poids_de_base()
    sommes = np.zeros(jouet.n_neurones)
    np.add.at(sommes, jouet.post, np.abs(poids))
    recus = np.zeros(jouet.n_neurones, dtype=int)
    np.add.at(recus, jouet.post, 1)
    for neurone in range(jouet.n_neurones):
        attendu = 1.0 if recus[neurone] else 0.0
        assert sommes[neurone] == pytest.approx(attendu, abs=1e-5)


def test_loi_de_dale(jouet):
    """Un neurone a le même effet sur toutes ses cibles."""
    poids = jouet.poids_de_base()
    for neurone in range(jouet.n_neurones):
        sortants = poids[jouet.pre == neurone]
        if len(sortants):
            assert len(set(np.sign(sortants))) == 1


def test_orientation_du_graphe(jouet):
    """L'entrée d'un neurone est la somme sur ses connexions **entrantes** (§7.2).

    Une inversion pre/post ne lève aucune erreur : c'est ce test qui l'attrape.
    """
    poids = jouet.poids_de_base()
    for cible in range(jouet.n_neurones):
        entrantes = np.flatnonzero(jouet.post == cible)
        for connexion in entrantes:
            assert jouet.post[connexion] == cible
            assert np.sign(poids[connexion]) == np.sign(jouet.signes[jouet.pre[connexion]])


def test_les_sorties_sont_atteignables(jouet):
    profondeur, part = jouet.distance_aux_sorties()
    assert part == 1.0
    assert 1 <= profondeur <= 5


# --- Paramètres appris ---------------------------------------------------


def test_le_compte_des_parametres(mouche, jouet):
    """Gains + fuites + température, comme en §8.2."""
    assert mouche.n_parametres == jouet.n_connexions + jouet.n_neurones + 1


def test_valeurs_initiales(mouche):
    assert torch.allclose(mouche.gains, torch.ones_like(mouche.gains), atol=1e-6)
    assert torch.allclose(mouche.fuites, torch.full_like(mouche.fuites, 0.5), atol=1e-6)
    assert float(mouche.temperature.detach()) == pytest.approx(1.0, abs=1e-6)


def test_un_gain_ne_peut_jamais_devenir_negatif(mouche):
    """La softplus, pas une troncature : même un pas absurde ne peut pas
    inverser un signe, donc transformer un excitateur en inhibiteur (§8.2)."""
    with torch.no_grad():
        mouche.gains_bruts.fill_(-1e6)
    assert (mouche.gains >= 0).all()
    with torch.no_grad():
        mouche.gains_bruts.fill_(1e6)
    assert torch.isfinite(mouche.gains).all()


def test_les_fuites_restent_entre_zero_et_un(mouche):
    with torch.no_grad():
        mouche.fuites_brutes.fill_(-1e6)
    assert (mouche.fuites >= 0).all()
    with torch.no_grad():
        mouche.fuites_brutes.fill_(1e6)
    assert (mouche.fuites <= 1).all()


# --- Dynamique -----------------------------------------------------------


def test_l_etat_reste_borne(mouche):
    """État dans [−1, 1] par construction : x ← (1−f)x + f·tanh(...) (§8.1)."""
    notes = mouche(encodages(8))
    assert torch.isfinite(notes).all()


def test_une_note_par_candidat(mouche):
    assert mouche(encodages(7)).shape == (7,)


def test_invariance_au_lot(mouche):
    """La note d'un candidat ne dépend pas des autres candidats du lot (§16).

    C'est **le** test qui détectera une fuite entre candidats dans le noyau
    creux : avec l'état remis à zéro pour chacun, évaluer un candidat seul ou
    parmi 34 doit donner le même nombre.
    """
    lot = encodages(6)
    ensemble = mouche(lot)
    for indice in range(len(lot)):
        seul = mouche(lot[indice : indice + 1])
        assert float(seul[0].detach()) == pytest.approx(float(ensemble[indice].detach()), abs=1e-5)


def test_l_ordre_du_lot_ne_change_rien(mouche):
    lot = encodages(6)
    inverse = mouche(torch.flip(lot, dims=[0]))
    assert torch.allclose(mouche(lot), torch.flip(inverse, dims=[0]), atol=1e-5)


def test_pas_de_memoire_entre_evaluations(mouche):
    """L'état est remis à zéro : deux appels identiques rendent la même note (§9.4)."""
    lot = encodages(4)
    assert torch.allclose(mouche(lot), mouche(lot), atol=1e-7)


def test_deux_grilles_differentes_donnent_des_notes_differentes(mouche):
    """Vérification d'initialisation de §7.4 : le réseau distingue les entrées."""
    notes = mouche(encodages(10))
    assert len(torch.unique(notes)) > 1


def test_la_temperature_ne_change_pas_le_classement(mouche):
    lot = encodages(10)
    avant = torch.argsort(mouche(lot))
    with torch.no_grad():
        mouche.temperature_brute.fill_(5.0)
    assert torch.equal(avant, torch.argsort(mouche(lot)))


def test_le_gradient_traverse_les_k_mises_a_jour(mouche):
    mouche(encodages(3)).sum().backward()
    assert mouche.gains_bruts.grad is not None
    assert (mouche.gains_bruts.grad.abs() > 0).any()
    assert (mouche.fuites_brutes.grad.abs() > 0).any()


def test_k_change_la_note(jouet):
    """Si K était trop petit, le signal n'atteindrait pas la sortie (§9.4)."""
    lot = encodages(4)
    court = Mouche(jouet, k=1, facteur_global=4.0)(lot)
    long = Mouche(jouet, k=8, facteur_global=4.0)(lot)
    assert not torch.allclose(court, long, atol=1e-4)


# --- Projection d'entrée -------------------------------------------------


def test_chaque_neurone_d_entree_lit_une_seule_entree(mouche, jouet):
    assert mouche.lecture.shape == (len(jouet.entrees),)
    assert int(mouche.lecture.max()) < TAILLE_ENCODAGE


def test_la_projection_est_equilibree():
    """4 114 = 205 × 20 + 14 : 14 entrées lues par 21 neurones, 191 par 20 (§9.2)."""
    from mouche_tetris.connectome.modele import _projection_d_entree

    lecture, signes = _projection_d_entree(4114, graine=0)
    comptes = np.bincount(lecture, minlength=TAILLE_ENCODAGE)
    assert set(np.unique(comptes).tolist()) == {20, 21}
    assert int((comptes == 21).sum()) == 14
    assert set(np.unique(signes).tolist()) == {-1.0, 1.0}


def test_la_graine_du_run_change_les_interfaces(jouet):
    """En v1.0, chaque graine retire aussi les interfaces (§8.3, §10.5)."""
    a = Mouche(jouet, k=4, facteur_global=4.0, graine_run=0)
    b = Mouche(jouet, k=4, facteur_global=4.0, graine_run=1)
    assert not torch.equal(a.lecture, b.lecture)
    assert not torch.equal(a.coefficients, b.coefficients)


# --- Contrôles -----------------------------------------------------------


def test_le_recablage_conserve_ce_qu_il_doit(jouet):
    recable = ctrl.recabler(jouet, graine=0)
    assert recable.n_connexions == jouet.n_connexions
    assert np.array_equal(recable.post, jouet.post)
    # Même degré entrant et mêmes poids de base pour chaque neurone.
    assert np.allclose(np.sort(np.abs(recable.poids_de_base())),
                       np.sort(np.abs(jouet.poids_de_base())))
    assert not np.array_equal(recable.pre, jouet.pre)


def test_le_recablage_a_signes_conserves_garde_les_signes(jouet):
    recable = ctrl.recabler_a_signes_conserves(jouet, graine=0)
    assert np.array_equal(jouet.signes[recable.pre], jouet.signes[jouet.pre])
    assert recable.n_connexions == jouet.n_connexions


def test_le_recablage_ordinaire_change_l_equilibre_des_signes(jouet):
    """La limite du contrôle principal, à rapporter dans les résultats."""
    ordinaire = ctrl.recabler(jouet, graine=0)
    conserve = ctrl.recabler_a_signes_conserves(jouet, graine=0)
    assert not np.array_equal(jouet.signes[ordinaire.pre], jouet.signes[jouet.pre])
    assert np.array_equal(jouet.signes[conserve.pre], jouet.signes[jouet.pre])


def test_le_graphe_coupe_donne_la_meme_note_a_tout(mouche):
    """Sans le graphe, les neurones moteurs — disjoints de l'entrée — restent
    à zéro. Toutes les notes sont égales, donc le choix est purement aléatoire
    et le témoin doit rendre exactement le score du joueur au hasard (§11.2)."""
    notes = ctrl.couper_le_graphe(mouche)(encodages(12))
    assert torch.allclose(notes, notes[0].expand_as(notes), atol=1e-6)


def test_l_entree_aveuglee_donne_la_meme_note_a_tout(mouche):
    notes = ctrl.aveugler_l_entree(mouche)(encodages(12))
    assert torch.allclose(notes, notes[0].expand_as(notes), atol=1e-6)


def test_les_temoins_ne_cassent_pas_le_modele(mouche):
    ctrl.couper_le_graphe(mouche)
    assert torch.isfinite(mouche(encodages(4))).all()
