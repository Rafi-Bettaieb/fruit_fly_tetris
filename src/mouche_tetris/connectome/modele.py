"""La mouche : le modèle à taux contraint par le connectome (conception.md §8, §9).

Chaque neurone porte un état entre −1 et 1, nul au départ de chaque évaluation.
À chaque mise à jour :

    état ← (1 − fuite) × état + fuite × tanh(somme des entrées)

Deux sortes d'entrées s'additionnent : les **synaptiques**, pour chaque connexion
entrante — état du présynaptique × signe × poids de base × gain appris × facteur
global — et la **sensorielle**, réservée aux 4 114 neurones d'entrée.

**Ce qui s'entraîne, et rien d'autre** (§8.2) :

| Paramètre | Nombre | Contrainte | Départ |
|---|---|---|---|
| Gain par connexion | 25 563 197 | softplus, donc positif | 1 |
| Fuite par neurone | 165 122 | sigmoïde, donc dans ]0, 1[ | 0,5 |
| Température | 1 | softplus | 1 |

Soit 25 728 320. Les contraintes passent par un reparamétrage et jamais par une
troncature après coup : **aucun gain ne peut devenir négatif**, donc aucun signe
ne peut s'inverser, quelle que soit la taille du pas d'apprentissage.

**Tout le signal passe par le graphe** (§9.3). La projection d'entrée et le
vecteur de sortie sont aléatoires, figés pour tout un run, et les populations
d'entrée et de sortie sont disjointes. Les témoins « graphe coupé » et « entrée
aveuglée » doivent retomber au niveau du hasard : s'ils font mieux, un raccourci
transporte le signal hors du connectome.

**La température ne change pas le jeu.** Multiplier toutes les notes d'une pièce
par un nombre positif ne change pas leur classement : elle n'agit que sur la
perte et sur les probabilités affichées dans la démo.

---

**Le produit synaptique passe par `noyau.py`**, repris de flyhard avec le
correctif de fly-self-driving, comme l'exige §14.2. La raison est mesurée sur ce
projet : une implémentation directe, qui matérialise un message par connexion,
dépasse les 4 Go de VRAM **dès 2 évaluations** — là où le protocole en demande
40 par pas. Et `torch.sparse.mm`, apparemment plus économe, calcule le gradient
des valeurs en passant par un intermédiaire dense de 165 122 × 165 122, soit
101,57 Gio.

Les arêtes sont donc rangées en CSR par neurone postsynaptique dès la
construction, et l'état circule en `(neurones, lot)` — la disposition que le
noyau attend.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..encodage import TAILLE_ENCODAGE
from .graphe import Graphe
from .noyau import ProduitCreuxParArete, csr_depuis_aretes

GAIN_INITIAL = 1.0
FUITE_INITIALE = 0.5
TEMPERATURE_INITIALE = 1.0


def _softplus_inverse(valeur: float) -> float:
    """Le paramètre brut dont la softplus rend `valeur` — ln(e^v − 1)."""
    return float(np.log(np.expm1(valeur)))


class Mouche(nn.Module):
    """Le connectome comme noteur de grilles."""

    def __init__(
        self,
        graphe: Graphe,
        k: int,
        facteur_global: float,
        graine_run: int = 0,
    ) -> None:
        super().__init__()
        self.graphe = graphe
        self.k = k
        self.facteur_global = facteur_global

        # --- Figé : la topologie et les poids mesurés, en structure CSR ---
        # Les arêtes sont réordonnées une fois par neurone postsynaptique, pour
        # que le noyau creux (§14.2) n'ait rien à permuter à chaque mise à jour.
        crow, col, rows, ordre = csr_depuis_aretes(graphe.pre, graphe.post, graphe.n_neurones)
        self.register_buffer("crow", torch.from_numpy(crow))
        self.register_buffer("col", torch.from_numpy(col))
        self.register_buffer("rows", torch.from_numpy(rows))
        self.register_buffer("poids_de_base", torch.from_numpy(graphe.poids_de_base()[ordre]))

        # --- Figé : les interfaces, tirées une fois avec la graine du run ---
        lecture, signe_entree = _projection_d_entree(len(graphe.entrees), graine_run)
        self.register_buffer("entrees", torch.from_numpy(graphe.entrees))
        self.register_buffer("lecture", torch.from_numpy(lecture))
        self.register_buffer("signe_entree", torch.from_numpy(signe_entree))
        self.register_buffer("sorties", torch.from_numpy(graphe.sorties))
        self.register_buffer("coefficients", torch.from_numpy(_vecteur_de_sortie(
            len(graphe.sorties), graine_run)))

        # --- Appris ---
        self.gains_bruts = nn.Parameter(
            torch.full((graphe.n_connexions,), _softplus_inverse(GAIN_INITIAL))
        )
        self.fuites_brutes = nn.Parameter(torch.zeros(graphe.n_neurones))  # sigmoïde(0) = 0,5
        self.temperature_brute = nn.Parameter(
            torch.tensor(_softplus_inverse(TEMPERATURE_INITIALE))
        )

        # Mis à 0 par le témoin « graphe coupé », à 0 par « entrée aveuglée ».
        self.echelle_des_gains = 1.0
        self.echelle_d_entree = 1.0

    # --- Paramètres contraints ------------------------------------------

    @property
    def gains(self) -> torch.Tensor:
        return nn.functional.softplus(self.gains_bruts) * self.echelle_des_gains

    @property
    def fuites(self) -> torch.Tensor:
        return torch.sigmoid(self.fuites_brutes)

    @property
    def temperature(self) -> torch.Tensor:
        return nn.functional.softplus(self.temperature_brute)

    @property
    def n_parametres(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --- Dynamique -------------------------------------------------------

    def entree_sensorielle(self, encodages: torch.Tensor) -> torch.Tensor:
        """Ce que chaque neurone d'entrée lit de la grille (§9.2).

        Chacun est affecté à une seule des 205 entrées et reçoit sa valeur
        multipliée par son signe. Rien d'autre n'entre dans le réseau.
        """
        return encodages[:, self.lecture] * self.signe_entree * self.echelle_d_entree

    def etats(self, encodages: torch.Tensor) -> torch.Tensor:
        """L'état des neurones après K mises à jour, forme (lot, n_neurones).

        Sert au réglage du facteur global et à la mesure de la participation
        (§7.4, §7.6), et alimente le nuage de neurones de la démo (§13.2).
        """
        if encodages.dim() == 1:
            encodages = encodages.unsqueeze(0)
        lot = encodages.shape[0]

        # Le noyau travaille en (neurones, lot) : c'est la disposition qui
        # permet au produit creux de lire une colonne d'états par arête.
        etat = torch.zeros(self.graphe.n_neurones, lot, device=encodages.device)
        sensorielle = self.entree_sensorielle(encodages).T
        valeurs = self.poids_de_base * self.gains * self.facteur_global
        fuite = self.fuites[:, None]

        for _ in range(self.k):
            entrees = ProduitCreuxParArete.apply(valeurs, self.crow, self.col, self.rows, etat)
            entrees = entrees.index_add(0, self.entrees, sensorielle)
            etat = (1 - fuite) * etat + fuite * torch.tanh(entrees)

        return etat.T

    def forward(self, encodages: torch.Tensor) -> torch.Tensor:
        """Une note par encodage, après K mises à jour depuis l'état nul."""
        moteurs = self.etats(encodages)[:, self.sorties]
        return (moteurs * self.coefficients).sum(dim=1) * self.temperature


def _projection_d_entree(n_entrees: int, graine: int) -> tuple[np.ndarray, np.ndarray]:
    """Affecte chaque neurone sensoriel à une entrée, avec un signe (§9.2).

    L'affectation est aléatoire mais **équilibrée** : avec 4 114 neurones pour
    205 entrées, 4 114 = 205 × 20 + 14, donc 14 entrées tirées au hasard sont
    lues par 21 neurones et les 191 autres par 20. Un tirage libre laisserait
    des entrées lues par 12 neurones et d'autres par 30 — certaines cases de la
    grille compteraient deux fois plus que d'autres, sans raison.
    """
    generateur = np.random.default_rng(graine)
    base, reste = divmod(n_entrees, TAILLE_ENCODAGE)
    comptes = np.full(TAILLE_ENCODAGE, base, dtype=np.int64)
    comptes[generateur.choice(TAILLE_ENCODAGE, size=reste, replace=False)] += 1
    lecture = np.repeat(np.arange(TAILLE_ENCODAGE), comptes)
    generateur.shuffle(lecture)
    signes = generateur.choice([1.0, -1.0], size=n_entrees).astype(np.float32)
    return lecture.astype(np.int64), signes


def _vecteur_de_sortie(n_sorties: int, graine: int) -> np.ndarray:
    """Les coefficients figés de la lecture motrice (§9.3).

    Loi normale d'écart-type 1/√n : la note garde ainsi un ordre de grandeur
    comparable à celui des états, quel que soit le nombre de neurones moteurs.
    """
    generateur = np.random.default_rng(graine + 1)
    return generateur.normal(0.0, 1.0 / np.sqrt(n_sorties), size=n_sorties).astype(np.float32)
