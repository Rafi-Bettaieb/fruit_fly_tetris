"""Le graphe : topologie, signes, normalisation (conception.md §7).

165 122 neurones tracés, 25 563 197 connexions dirigées. Ici, la structure qui
les porte et les règles qui la préparent — la lecture des fichiers MaleCNS
viendra dans `acquisition`.

**Ce qui est figé, définitivement** (§8.3) : la topologie, les signes, les
nombres de synapses et leur normalisation, le facteur global. Aucune connexion
n'est jamais ajoutée, retirée ni déplacée.

**Normalisation** (§7.4) : poids de base = signe × nombre de synapses, divisé
par le total des synapses reçues par le neurone postsynaptique. La somme des
valeurs absolues des poids entrants de chaque neurone vaut donc 1 — sauf pour
les neurones sans connexion entrante, dont l'entrée synaptique reste nulle.

**Orientation** : les connexions vont du présynaptique au postsynaptique, et
l'entrée d'un neurone est la somme sur ses connexions **entrantes**. Une erreur
d'orientation ne produit aucun message d'erreur : elle est testée sur le graphe
jouet (§7.2, §16).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Loi de Dale, convention de §7.3 : seuls GABA, glutamate et histamine inhibent.
SIGNES = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 1,
    "serotonine": 1,
    "octopamine": 1,
    "tyramine": 1,
    "inconnu": 1,
}


@dataclass(frozen=True)
class Graphe:
    """Un graphe dirigé et signé, prêt pour le modèle.

    `pre` et `post` ont la même longueur : une entrée par connexion. `signes`
    donne le signe de chaque **neurone** (loi de Dale : un neurone a le même
    effet sur toutes ses cibles), `synapses` le nombre de contacts de chaque
    connexion.
    """

    pre: np.ndarray
    post: np.ndarray
    synapses: np.ndarray
    signes: np.ndarray
    n_neurones: int
    entrees: np.ndarray
    """Indices des neurones sensoriels — 4 114 dans le lobe optique."""

    sorties: np.ndarray
    """Indices des neurones moteurs — 708 dans la corde nerveuse ventrale."""

    def __post_init__(self) -> None:
        if len(self.pre) != len(self.post) != len(self.synapses):
            raise ValueError("pre, post et synapses doivent avoir la même longueur")
        if len(self.signes) != self.n_neurones:
            raise ValueError("un signe par neurone (loi de Dale)")
        if set(self.entrees.tolist()) & set(self.sorties.tolist()):
            raise ValueError(
                "les populations d'entrée et de sortie doivent être disjointes (§9.3) : "
                "sinon un signal pourrait rejoindre la sortie sans traverser le graphe"
            )

    @property
    def n_connexions(self) -> int:
        return len(self.pre)

    @property
    def part_inhibitrice(self) -> float:
        """Part des connexions dont le neurone source est inhibiteur (§7.6)."""
        return float(np.mean(self.signes[self.pre] < 0))

    def poids_de_base(self) -> np.ndarray:
        """Signe × synapses, normalisé par neurone postsynaptique (§7.4)."""
        recues = np.zeros(self.n_neurones, dtype=np.float64)
        np.add.at(recues, self.post, self.synapses)
        # Un neurone sans connexion entrante garde une entrée synaptique nulle :
        # aucune division, donc aucune valeur infinie qui se propagerait.
        recues[recues == 0] = 1.0
        return (self.signes[self.pre] * self.synapses / recues[self.post]).astype(np.float32)

    def sans_connexion_entrante(self) -> int:
        recues = np.zeros(self.n_neurones, dtype=np.int64)
        np.add.at(recues, self.post, 1)
        return int((recues == 0).sum())

    def distance_aux_sorties(self, couverture: float = 0.9) -> tuple[int, float]:
        """La distance D de §7.6, et la part de neurones moteurs atteignables.

        D est le plus petit nombre de sauts qui atteint `couverture` des neurones
        moteurs en partant des neurones d'entrée. C'est lui qui fixe le temps de
        réflexion K = D + 4 (§9.4) : si K est plus petit que D, le signal n'a
        même pas le temps de parvenir à la sortie, et la mouche échouerait pour
        une raison sans rapport avec le câblage.

        Si moins de `couverture` des neurones moteurs sont atteignables, D est la
        profondeur qui atteint 90 % des **atteignables**, et la part réelle est
        rapportée avec.
        """
        voisins: dict[int, list[int]] = {}
        for source, cible in zip(self.pre.tolist(), self.post.tolist(), strict=True):
            voisins.setdefault(source, []).append(cible)

        cibles = set(self.sorties.tolist())
        vus = set(self.entrees.tolist())
        front = list(vus)
        atteintes: set[int] = cibles & vus
        profondeur = 0
        paliers: list[tuple[int, int]] = [(0, len(atteintes))]

        while front:
            profondeur += 1
            suivant = []
            for neurone in front:
                for cible in voisins.get(neurone, ()):
                    if cible not in vus:
                        vus.add(cible)
                        suivant.append(cible)
                        if cible in cibles:
                            atteintes.add(cible)
            front = suivant
            paliers.append((profondeur, len(atteintes)))

        part = len(atteintes) / len(cibles)
        seuil = couverture * len(atteintes)
        for profondeur, combien in paliers:
            if combien >= seuil and combien > 0:
                return profondeur, part
        return profondeur, part


def _construire(
    n_neurones: int, n_entrees: int, n_sorties: int, degre: int, graine: int
) -> Graphe:
    generateur = np.random.default_rng(graine)
    entrees = np.arange(0, n_entrees)
    sorties = np.arange(n_neurones - n_sorties, n_neurones)
    milieu = np.arange(n_entrees, n_neurones - n_sorties)

    pre: list[int] = []
    post: list[int] = []
    # Entrée → milieu → sortie, pour que les neurones moteurs soient bien
    # atteignables, plus des connexions internes pour donner de la matière.
    for source in entrees:
        for cible in generateur.choice(milieu, size=degre, replace=False):
            pre.append(int(source))
            post.append(int(cible))
    vers = np.concatenate([milieu, sorties])
    for source in milieu:
        for cible in generateur.choice(vers, size=degre, replace=False):
            if cible != source:
                pre.append(int(source))
                post.append(int(cible))
    for source in milieu[: max(4, n_sorties)]:
        for cible in sorties:
            pre.append(int(source))
            post.append(int(cible))

    return Graphe(
        pre=np.array(pre, dtype=np.int64),
        post=np.array(post, dtype=np.int64),
        synapses=generateur.integers(1, 20, size=len(pre)).astype(np.float64),
        signes=generateur.choice([1, -1], size=n_neurones, p=[0.7, 0.3]).astype(np.int64),
        n_neurones=n_neurones,
        entrees=entrees,
        sorties=sorties,
    )


def graphe_jouet(n_neurones: int = 20, graine: int = 0) -> Graphe:
    """Le graphe de 20 neurones sur lequel tourne l'intégration continue (§16).

    Il est **construit**, jamais téléchargé : la CI ne doit pas tirer 1,2 Go de
    MaleCNS à chaque envoi de code. Il reproduit les propriétés qui comptent
    pour ce qu'on lui demande — populations d'entrée et de sortie disjointes,
    sortie atteignable, mélange d'excitateurs et d'inhibiteurs, nombres de
    synapses variés.

    **Il ne sert pas à vérifier qu'une mouche apprend**, et il ne le peut pas :
    avec 4 neurones sensoriels pour 205 entrées, il ne lit que 4 cases de la
    grille et ne distingue donc quasiment aucun candidat. Pour la chaîne
    complète, voir `petit_connectome`.
    """
    return _construire(n_neurones, n_entrees=4, n_sorties=4, degre=3, graine=graine)


def petit_connectome(
    n_neurones: int = 1200, n_entrees: int = 410, n_sorties: int = 82, graine: int = 0
) -> Graphe:
    """Un connectome réduit, assez grand pour que la chaîne complète tienne debout.

    Il respecte la seule contrainte qui empêche le graphe jouet d'apprendre :
    **au moins 205 neurones d'entrée**, pour que chaque case de la grille soit
    lue par au moins un neurone. Ici 410, soit deux par entrée — contre vingt
    dans le connectome réel, où 4 114 neurones lisent 205 entrées.

    Les proportions suivent celles du MaleCNS : 2,5 % de neurones d'entrée
    (4 114 sur 165 122) et 0,4 % de neurones moteurs (708). Il sert à valider
    l'enchaînement complet — encodage, K mises à jour, lecture motrice, perte,
    rétropropagation, jeu — avant que la même chaîne ne coûte des heures de GPU.
    """
    return _construire(n_neurones, n_entrees, n_sorties, degre=4, graine=graine)
