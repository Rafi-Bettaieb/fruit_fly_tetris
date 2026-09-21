"""Le noyau creux, confronté à un calcul naïf (conception.md §14.2).

Le document interdit d'écrire un nouveau noyau et impose celui de flyhard. Il
impose donc aussi de vérifier qu'il calcule bien ce qu'on croit : un noyau qui
se tromperait de 1 % sur le gradient ne lèverait aucune erreur et produirait un
entraînement silencieusement faux.

La référence est ici l'implémentation directe — `gather`, multiplication,
`index_add` — celle qui est juste par construction mais qui dépasse les 4 Go dès
deux évaluations sur le graphe complet. Sur le graphe jouet, elle tient sans
peine et fait arbitre.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mouche_tetris.connectome import noyau
from mouche_tetris.connectome.graphe import graphe_jouet


@pytest.fixture
def jouet():
    return graphe_jouet()


@pytest.fixture
def csr(jouet):
    crow, col, rows, ordre = noyau.csr_depuis_aretes(jouet.pre, jouet.post, jouet.n_neurones)
    return (
        torch.from_numpy(crow),
        torch.from_numpy(col),
        torch.from_numpy(rows),
        torch.from_numpy(jouet.poids_de_base()[ordre]).double(),
    )


def produit_naif(valeurs, col, rows, etat, n):
    """L'implémentation directe : un message par arête, puis somme par cible."""
    sortie = torch.zeros(n, etat.shape[1], dtype=etat.dtype)
    return sortie.index_add(0, rows, valeurs[:, None] * etat[col])


# --- Conversion en CSR ---------------------------------------------------


def test_la_conversion_preserve_les_aretes(jouet):
    crow, col, rows, ordre = noyau.csr_depuis_aretes(jouet.pre, jouet.post, jouet.n_neurones)
    assert len(col) == jouet.n_connexions
    assert crow[0] == 0 and crow[-1] == jouet.n_connexions
    # Mêmes paires (pre, post), seulement réordonnées.
    avant = sorted(zip(jouet.pre.tolist(), jouet.post.tolist(), strict=True))
    apres = sorted(zip(col.tolist(), rows.tolist(), strict=True))
    assert avant == apres


def test_l_ordre_permute_bien_les_tableaux_par_arete(jouet):
    """`ordre` doit réaligner les synapses et les gains sur la structure CSR.

    S'il ne le faisait pas, chaque connexion porterait le poids d'une autre —
    sans qu'aucune erreur ne se produise.
    """
    crow, col, rows, ordre = noyau.csr_depuis_aretes(jouet.pre, jouet.post, jouet.n_neurones)
    assert (jouet.pre[ordre] == col).all()
    assert (jouet.post[ordre] == rows).all()


def test_la_matrice_est_orientee_post_pre(jouet):
    """Ligne = postsynaptique, colonne = présynaptique (§7.2).

    Une inversion ne lève aucune erreur : le réseau tournerait à l'envers.
    """
    crow, col, rows, _ = noyau.csr_depuis_aretes(jouet.pre, jouet.post, jouet.n_neurones)
    for neurone in range(jouet.n_neurones):
        debut, fin = int(crow[neurone]), int(crow[neurone + 1])
        assert all(int(r) == neurone for r in rows[debut:fin])
        attendues = sorted(jouet.pre[jouet.post == neurone].tolist())
        assert sorted(col[debut:fin].tolist()) == attendues


# --- Équivalence avec le calcul naïf -------------------------------------


@pytest.mark.parametrize("lot", [1, 3, 8])
def test_le_passage_avant_est_identique(csr, jouet, lot):
    crow, col, rows, valeurs = csr
    etat = torch.randn(jouet.n_neurones, lot, dtype=torch.float64)
    rapide = noyau.ProduitCreuxParArete.apply(valeurs, crow, col, rows, etat)
    naif = produit_naif(valeurs, col, rows, etat, jouet.n_neurones)
    assert torch.allclose(rapide, naif, atol=1e-10)


@pytest.mark.parametrize("lot", [1, 4])
def test_les_gradients_sont_identiques(csr, jouet, lot):
    """Le test qui compte : c'est le gradient qui entraîne le modèle."""
    crow, col, rows, base = csr
    etat_source = torch.randn(jouet.n_neurones, lot, dtype=torch.float64)

    v1 = base.clone().requires_grad_(True)
    e1 = etat_source.clone().requires_grad_(True)
    perte1 = (noyau.ProduitCreuxParArete.apply(v1, crow, col, rows, e1) ** 2).sum()
    perte1.backward()

    v2 = base.clone().requires_grad_(True)
    e2 = etat_source.clone().requires_grad_(True)
    perte2 = (produit_naif(v2, col, rows, e2, jouet.n_neurones) ** 2).sum()
    perte2.backward()

    assert torch.allclose(perte1, perte2, atol=1e-10)
    assert torch.allclose(v1.grad, v2.grad, atol=1e-10), "gradient des gains"
    assert torch.allclose(e1.grad, e2.grad, atol=1e-10), "gradient des états"


def test_le_decoupage_ne_change_pas_le_gradient(csr, jouet, monkeypatch):
    """Le découpage borne la mémoire ; il ne doit rien changer au résultat.

    C'est ce qui autorise à adapter la tranche à une carte de 4 Go (§14.1).
    """
    crow, col, rows, base = csr
    etat = torch.randn(jouet.n_neurones, 4, dtype=torch.float64)
    gradients = []
    for tranche in (7, 64, 10_000):
        monkeypatch.setattr(noyau, "TRANCHE_CPU", tranche)
        valeurs = base.clone().requires_grad_(True)
        (noyau.ProduitCreuxParArete.apply(valeurs, crow, col, rows, etat) ** 2).sum().backward()
        gradients.append(valeurs.grad.clone())
    for autre in gradients[1:]:
        assert torch.allclose(gradients[0], autre, atol=1e-12)


def test_la_taille_de_tranche_respecte_le_budget():
    for lot in (1, 10, 40, 200, 5000):
        tranche = noyau.taille_de_tranche(lot)
        assert noyau.TRANCHE_MIN <= tranche <= noyau.TRANCHE_MAX
        tient = tranche * lot * 4 <= noyau.BUDGET_TRANCHE_OCTETS
        assert tient or tranche == noyau.TRANCHE_MIN


# --- Le cache ------------------------------------------------------------


def test_le_cache_ne_garde_qu_une_entree(csr, jouet):
    """Sur le graphe complet il pèse 400 Mo : deux entrées dépassent les 4 Go.

    Le bug a été vu en vrai, en enchaînant trois valeurs de K dans le même
    processus — `del modele` ne libérait rien, le dictionnaire tenant la
    référence.
    """
    noyau.vider_le_cache()
    crow, col, rows, valeurs = csr
    etat = torch.randn(jouet.n_neurones, 2, dtype=torch.float64, requires_grad=True)
    for _ in range(3):
        # Des tenseurs neufs à chaque tour : nouvelles adresses, nouvelle clé.
        c2, l2, r2 = crow.clone(), col.clone(), rows.clone()
        v2 = valeurs.clone().requires_grad_(True)
        noyau.ProduitCreuxParArete.apply(v2, c2, l2, r2, etat).sum().backward()
        assert len(noyau._CACHE_TRANSPOSEE) <= 1
    noyau.vider_le_cache()
    assert len(noyau._CACHE_TRANSPOSEE) == 0


def test_la_structure_transposee_est_correcte(csr, jouet):
    crow, col, rows, _ = csr
    noyau.vider_le_cache()
    crow_t, col_t, permutation = noyau._structure_transposee(crow, col, rows, jouet.n_neurones)
    assert crow_t[-1] == jouet.n_connexions
    # La transposée groupe par présynaptique : mêmes paires, rôles échangés.
    attendu = sorted(zip(jouet.post.tolist(), jouet.pre.tolist(), strict=True))
    obtenu = sorted(zip(col_t.tolist(), np.repeat(
        np.arange(jouet.n_neurones), np.diff(crow_t.numpy())).tolist(), strict=True))
    assert attendu == obtenu
    noyau.vider_le_cache()
