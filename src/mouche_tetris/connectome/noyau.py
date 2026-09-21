"""Le noyau creux : produit matrice creuse × états, sans matérialiser les messages.

**Ce fichier est repris de flyhard, avec le correctif de fly-self-driving.**
Le document l'exige (§14.2) : « Ce noyau est celui de flyhard, avec le
correctif de fly-self-driving. On n'en écrit pas un nouveau. »

- `flyhard` — Mark Unthank, licence MIT, commit `328906f`
- `fly-self-driving` — Pure Reason Inc., licence MIT, commit `3516a09`

Le problème qu'il résout, mesuré sur ce projet avant d'avoir lu leur code :
`torch.sparse.mm` calcule le gradient des valeurs d'une matrice CSR en passant
par un intermédiaire **dense** de 165 122 × 165 122, soit **101,57 Gio** pour
une seule mise à jour. Le docstring de flyhard cite exactement le même chiffre.

La dérivée exacte à l'arête i ← j est `dot(dL/dY[i], X[j])` : elle se calcule
arête par arête, sans matrice dense. Le découpage en tranches borne la mémoire
temporaire à `tranche × lot` au lieu de `arêtes × lot`.

Le correctif de fly-self-driving ajoute le **cache de la structure transposée**.
cuSPARSE reconstruit une CSR transposée à chaque appel, ce qui coûte environ
sept fois le passage avant ; la garder une fois et n'y permuter que les valeurs
rend la rétropropagation 2,5 fois plus rapide.

Sans ce noyau, l'entraînement est impossible sur 4 Go : mesuré, l'implémentation
de référence de `modele.py` dépasse la mémoire dès **2 évaluations**, là où le
protocole en demande 40 par pas.
"""

from __future__ import annotations

import numpy as np
import torch

BUDGET_TRANCHE_OCTETS = 128 << 20
"""Mémoire temporaire que la rétropropagation s'autorise pour une tranche.

C'est le seul réglage adapté à cette machine, et il ne change **rien** aux
mathématiques : le gradient est le même, seul le découpage diffère. Le correctif
d'origine fixe la tranche à 8 388 608 arêtes, taille calibrée pour la carte de
48 Go sur laquelle il a été écrit. Sur 4 Go, avec 40 évaluations, cette tranche
demande 8,4 M × 40 × 4 octets = 1,34 Gio d'un coup — c'est elle, et non le
graphe, qui faisait dépasser la mémoire.
"""

TRANCHE_MIN = 65_536
TRANCHE_MAX = 8_388_608
TRANCHE_CPU = 262_144


def taille_de_tranche(lot: int) -> int:
    """Nombre d'arêtes par tranche, pour tenir dans le budget quel que soit le lot."""
    par_arete = max(1, lot) * 4
    return int(min(TRANCHE_MAX, max(TRANCHE_MIN, BUDGET_TRANCHE_OCTETS // par_arete)))

_CACHE_TRANSPOSEE: dict = {}


def vider_le_cache() -> None:
    """Libère la structure transposée. À appeler entre deux modèles."""
    _CACHE_TRANSPOSEE.clear()


def _structure_transposee(crow, col, rows, n):
    """Structure CSR du graphe transposé, mise en cache.

    **Le cache ne garde qu'une seule entrée.** Il pèse environ 400 Mo sur le
    graphe complet, et la clé est une adresse mémoire : construire un second
    modèle — pour une autre valeur de K, ou pour la mouche recâblée — crée une
    nouvelle entrée pendant que l'ancienne reste référencée par ce dictionnaire.
    Deux modèles suffisaient alors à dépasser les 4 Go, et `del modele` n'y
    changeait rien. On n'entraîne jamais deux graphes à la fois : garder la
    dernière entrée suffit, et borne la mémoire.
    """
    cle = (col.data_ptr(), crow.data_ptr(), str(col.device))
    if cle not in _CACHE_TRANSPOSEE:
        _CACHE_TRANSPOSEE.clear()
        permutation = torch.argsort(col, stable=True)
        comptes = torch.bincount(col, minlength=n)
        crow_t = torch.zeros(n + 1, dtype=crow.dtype, device=col.device)
        crow_t[1:] = torch.cumsum(comptes, 0)
        _CACHE_TRANSPOSEE[cle] = (crow_t, rows[permutation], permutation)
    return _CACHE_TRANSPOSEE[cle]


class ProduitCreuxParArete(torch.autograd.Function):
    """Produit matrice creuse × états, dérivé uniquement aux arêtes mesurées."""

    @staticmethod
    def forward(ctx, valeurs, crow, col, rows, etat):
        n = len(crow) - 1
        matrice = torch.sparse_csr_tensor(crow, col, valeurs, size=(n, n), check_invariants=False)
        ctx.save_for_backward(valeurs, crow, col, rows, etat)
        return torch.sparse.mm(matrice, etat)

    @staticmethod
    def backward(ctx, gradient_sortie):
        valeurs, crow, col, rows, etat = ctx.saved_tensors
        gradient_valeurs = torch.empty_like(valeurs) if ctx.needs_input_grad[0] else None
        if gradient_valeurs is not None:
            lot = gradient_sortie.shape[1]
            tranche = taille_de_tranche(lot) if valeurs.is_cuda else TRANCHE_CPU
            for debut in range(0, len(valeurs), tranche):
                fin = min(debut + tranche, len(valeurs))
                gradient_valeurs[debut:fin] = (
                    gradient_sortie[rows[debut:fin]] * etat[col[debut:fin]]
                ).sum(dim=1)

        gradient_etat = None
        if ctx.needs_input_grad[4]:
            n = len(crow) - 1
            if valeurs.is_cuda:
                crow_t, col_t, permutation = _structure_transposee(crow, col, rows, n)
                matrice_t = torch.sparse_csr_tensor(
                    crow_t, col_t, valeurs[permutation], size=(n, n), check_invariants=False
                )
                gradient_etat = torch.sparse.mm(matrice_t, gradient_sortie)
            else:
                matrice = torch.sparse_csr_tensor(
                    crow, col, valeurs, size=(n, n), check_invariants=False
                )
                gradient_etat = torch.sparse.mm(matrice.transpose(0, 1), gradient_sortie)
        return gradient_valeurs, None, None, None, gradient_etat


def csr_depuis_aretes(
    pre: np.ndarray, post: np.ndarray, n_neurones: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convertit une liste d'arêtes en structure CSR groupée par postsynaptique.

    La matrice est **(post, pre)** : ligne = neurone postsynaptique, colonne =
    présynaptique, conformément à §7.2 — l'entrée d'un neurone est la somme sur
    ses connexions entrantes. Inverser les deux ne lève aucune erreur, d'où le
    test d'orientation sur le graphe jouet (§16).

    Rend `(crow, col, rows, ordre)`. `ordre` permute les tableaux par arête —
    synapses, gains — dans l'ordre CSR, une fois pour toutes à la construction.
    """
    ordre = np.argsort(post, kind="stable")
    rows = post[ordre]
    col = pre[ordre]
    comptes = np.bincount(rows, minlength=n_neurones)
    crow = np.zeros(n_neurones + 1, dtype=np.int64)
    np.cumsum(comptes, out=crow[1:])
    return crow, col, rows, ordre
