"""L'interface commune à toutes les conditions comparées (conception.md §11.2).

Le projet compare dix conditions : la mouche, la mouche recâblée, le recâblage à
signes conservés, le réservoir, les élèves linéaire / MLP / convolutif, les deux
témoins, le joueur au hasard et l'expert.

Elles n'ont rien en commun — l'une traverse 25,6 millions de connexions sur GPU,
l'autre applique quatre poids — sauf **ceci** : on leur donne les grilles
candidates d'une pièce, elles rendent une note par candidat. Tout le reste du
projet (évaluation appariée, enregistrement, serveur de diffusion) ne parle qu'à
cette interface et ignore ce qu'il pilote.

C'est la décision d'architecture centrale. Elle garantit mécaniquement que toutes
les conditions sont mesurées par exactement le même code, ce dont dépend la
comparabilité de la section 11.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Noteur(Protocol):
    """Ce que toute condition du projet doit savoir faire.

    On lui passe les candidats d'une pièce, il rend une note par candidat.

    Contrat, valable pour toutes les implémentations sans exception :

    - le retour a la forme (n_candidats,) : une note réelle par candidat, plus
      la note est haute, meilleur est le candidat ;
    - **aucun état n'est conservé d'un appel à l'autre.** Pour la mouche, l'état
      du réseau est remis à zéro pour chaque candidat (§9.4) ; pour les autres,
      la question ne se pose pas. Deux appels identiques rendent des notes
      identiques ;
    - **la note d'un candidat ne dépend pas des autres candidats du lot.**
      Évaluer un candidat seul ou parmi 34 donne le même nombre. C'est testé
      (§16) : c'est la seule façon de détecter une fuite entre candidats dans
      le noyau creux ;
    - **un noteur fondé sur un modèle ne lit du candidat que son encodage**, via
      `encodage.encoder_lot`. Jamais sa rotation, sa colonne, ni la pièce dont
      il vient. L'expert, lui, a le droit de lire la grille directement : il
      voit exactement la même chose, seulement sans passer par les 205 entrées.

    Le paramètre est la liste des candidats plutôt que leurs encodages parce que
    l'expert note la grille avec quatre critères (§10.1) : le faire passer par
    un encodage en ±1 qu'il devrait décoder serait absurde.
    """

    nom: str

    def noter(self, candidats) -> np.ndarray:
        """Rend une note par candidat, dans l'ordre reçu."""
        ...


class Budget:
    """Ce qu'une condition a consommé pour arriver à son score (§3, §11.1).

    Publié à côté de chaque ligne du tableau de résultats. Sans lui, un écart
    entre deux modèles entraînés sur des budgets différents se lit comme un
    écart de capacité, ce qu'il n'est pas.
    """

    def __init__(
        self,
        situations_vues: int,
        mises_a_jour: int,
        parametres: int,
        graine_run: int,
    ) -> None:
        self.situations_vues = situations_vues
        self.mises_a_jour = mises_a_jour
        self.parametres = parametres
        self.graine_run = graine_run
