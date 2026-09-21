"""Les signes des neurones, loi de Dale (conception.md §7.3).

Un neurone a le même effet — excitateur ou inhibiteur — sur toutes ses cibles.
Le signe vient de la prédiction de neurotransmetteur, colonne `consensus_nt`,
avec la convention du document :

| Neurotransmetteur | Signe | Pourquoi |
|---|---|---|
| acétylcholine | + | |
| GABA, glutamate | − | convention de Shiu et al., Nature 2024 |
| histamine | − | neurotransmetteur des photorécepteurs, canaux chlorure |
| dopamine, sérotonine, octopamine, tyramine | + | |
| `unclear`, absent | + | comme dans les modèles du cerveau entier |

**Ce que cette convention donne réellement, mesuré :**

| Population | Part inhibitrice |
|---|---|
| Les 165 122 neurones tracés | 34,7 % |
| Les 4 114 neurones d'entrée | **100 %** |
| Les 708 neurones moteurs | 42,9 % |

**Les neurones d'entrée sont tous inhibiteurs**, sans exception : ce sont des
photorécepteurs. Toute l'information sur la grille entre donc dans le graphe par
de l'inhibition. Cela ne perd rien — la projection d'entrée donne à chaque
neurone sensoriel un signe aléatoire ±1, donc la moitié s'activent sur une case
occupée et l'autre moitié sur une case vide, et le signal reste bidirectionnel
en aval. Mais cela donne à ce tirage un rôle que le document ne lui prêtait pas :
c'est lui seul qui empêche l'entrée d'être un biais systématique.

**Plus de la moitié des signes de sortie sont inférés par défaut.** 54,9 % des
neurones moteurs ont un neurotransmetteur `unclear` et deviennent excitateurs
par convention, non par mesure. Le registre de §8.4 le dit : pour cette
population, le signe est surtout un choix de ce document.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .acquisition import BRUT, chemin

COLONNE_IDENTIFIANT = "body"
COLONNE_NEUROTRANSMETTEUR = "consensus_nt"

INHIBITEURS = frozenset({"gaba", "glutamate", "histamine"})
EXCITATEURS = frozenset(
    {"acetylcholine", "dopamine", "serotonin", "octopamine", "tyramine", "unclear"}
)

SIGNE_PAR_DEFAUT = 1
"""Un neurone absent du fichier de neurotransmetteurs — 502 des 165 122 — est
excitateur, comme les `unclear` : même convention, même statut d'inféré."""


def signe_du_neurotransmetteur(nom: str | None) -> int:
    """+1 ou −1, selon la convention de §7.3."""
    if nom is None:
        return SIGNE_PAR_DEFAUT
    return -1 if nom in INHIBITEURS else SIGNE_PAR_DEFAUT


def lire_les_signes(identifiants: np.ndarray, racine: Path = BRUT) -> np.ndarray:
    """Le signe de chaque neurone demandé, dans l'ordre reçu.

    `identifiants` sont des `bodyId` du fichier d'annotations ; le fichier de
    neurotransmetteurs les appelle `body` et en contient bien plus — 1,8 million
    de segments contre 165 122 neurones tracés. On ne garde que les demandés.
    """
    import pyarrow.feather as feather

    table = feather.read_table(chemin("neurotransmetteurs", racine), memory_map=True)
    corps = np.asarray(table.column(COLONNE_IDENTIFIANT).to_pylist(), dtype=np.int64)
    predits = np.asarray(table.column(COLONNE_NEUROTRANSMETTEUR).to_pylist(), dtype=object)

    par_identifiant = dict(zip(corps.tolist(), predits.tolist(), strict=True))
    return np.array(
        [signe_du_neurotransmetteur(par_identifiant.get(int(i))) for i in identifiants],
        dtype=np.int64,
    )


def composition(identifiants: np.ndarray, racine: Path = BRUT) -> dict[str, float]:
    """Répartition des neurotransmetteurs d'une population, pour les résultats."""
    import pyarrow.feather as feather

    table = feather.read_table(chemin("neurotransmetteurs", racine), memory_map=True)
    corps = np.asarray(table.column(COLONNE_IDENTIFIANT).to_pylist(), dtype=np.int64)
    predits = np.asarray(table.column(COLONNE_NEUROTRANSMETTEUR).to_pylist(), dtype=object)
    par_identifiant = dict(zip(corps.tolist(), predits.tolist(), strict=True))

    noms = [par_identifiant.get(int(i), "absent") for i in identifiants]
    valeurs, comptes = np.unique(np.array(noms, dtype=str), return_counts=True)
    total = len(noms)
    return {
        str(valeur): int(compte) / total
        for valeur, compte in zip(valeurs, comptes, strict=True)
    }
