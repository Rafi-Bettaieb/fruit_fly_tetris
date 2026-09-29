"""Le direct (§13.3), sans GPU : le joueur au hasard tient le rôle de la mouche.

Ce qui est vérifié ici, c'est le contrat avec la page — chaque coup diffusé
porte tout ce qu'il faut pour montrer la pièce apparaître, tourner, glisser et
tomber — et le rythme : le coup suivant n'arrive qu'une fois que la patte a
fini d'appuyer.
"""

from __future__ import annotations

import asyncio
import json

from mouche_tetris.modeles.hasard import Hasard
from mouche_tetris.serveur import diffusion
from mouche_tetris.tetris.touches import Touche, reconstituer


class _Spectateur:
    def __init__(self) -> None:
        self.recus: list[dict] = []

    async def send_text(self, texte: str) -> None:
        self.recus.append(json.loads(texte))


def _une_partie_courte(monkeypatch) -> tuple[list[dict], list[float]]:
    attentes: list[float] = []

    async def attendre(secondes):
        attentes.append(secondes)

    monkeypatch.setattr(diffusion.asyncio, "sleep", attendre)
    direct = diffusion.Direct(Hasard(), plafond=12)
    spectateur = _Spectateur()
    direct.spectateurs.add(spectateur)
    asyncio.run(direct._une_partie(diffusion.GRAINE_DEPART))
    return spectateur.recus, attentes


def test_chaque_coup_porte_sa_trajectoire(monkeypatch):
    recus, _ = _une_partie_courte(monkeypatch)
    coups = [m for m in recus if m["type"] == "coup"]
    assert coups
    for coup in coups:
        assert len(coup["trajectoire"]) == len(coup["touches"])
        assert coup["touches"][-1] == Touche.CHUTE.value
        attendues = reconstituer(coup["piece"], coup["choix"]["rotation"], coup["choix"]["colonne"])
        assert coup["touches"] == [t.value for t in attendues]


def test_le_coup_suivant_attend_la_fin_des_appuis(monkeypatch):
    """Sinon la pièce suivante arrive pendant que la patte appuie encore, et
    l'on ne voit jamais la rotation — le défaut qui a fait ajouter cette règle."""
    recus, attentes = _une_partie_courte(monkeypatch)
    coups = [m for m in recus if m["type"] == "coup"]
    for coup, attente in zip(coups, attentes, strict=False):
        assert attente >= diffusion.CADENCE
        assert attente >= diffusion.PAS_TOUCHE * len(coup["touches"])


class _HasardQuiMontre(Hasard):
    """Un joueur factice doté de la capacité facultative `activite`."""

    def activite(self, candidat):
        return {"k": 7, "n": 3, "etapes": "AAAAAAAAAAAAAAAAAAAAAAAAAAAA"}

    def description_du_nuage(self):
        return {"n": 3, "k": 7, "empreinte": "0" * 12}


def test_l_activite_part_avec_chaque_coup_et_retarde_le_suivant(monkeypatch):
    attentes: list[float] = []

    async def attendre(secondes):
        attentes.append(secondes)

    monkeypatch.setattr(diffusion.asyncio, "sleep", attendre)
    direct = diffusion.Direct(_HasardQuiMontre(), plafond=6)
    spectateur = _Spectateur()
    direct.spectateurs.add(spectateur)
    asyncio.run(direct._une_partie(diffusion.GRAINE_DEPART))
    coups = [m for m in spectateur.recus if m["type"] == "coup"]
    assert coups and all(coup["neurones"]["k"] == 7 for coup in coups)
    for coup, attente in zip(coups, attentes, strict=False):
        # La vague (7 mises à jour) passe avant les appuis : le coup suivant attend les deux.
        appuis = diffusion.PAS_TOUCHE * len(coup["touches"])
        assert attente >= 7 * diffusion.PAS_MISE_A_JOUR + appuis


def test_sans_activite_le_champ_reste_vide(monkeypatch):
    recus, _ = _une_partie_courte(monkeypatch)
    assert all(m.get("neurones") is None for m in recus if m["type"] == "coup")
