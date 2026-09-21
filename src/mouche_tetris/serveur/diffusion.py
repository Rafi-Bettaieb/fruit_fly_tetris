"""Le direct : la mouche joue, tout le monde regarde la même partie (§13.3).

**Une seule partie, diffusée à tous.** C'est la décision qui rend le direct
possible sur une carte de 4 Go : le serveur ne lance pas une partie par
visiteur, il en fait tourner **une**, en continu, et envoie le même état à tous
les navigateurs connectés. Le coût GPU est donc constant — 0,20 s par pièce —
qu'il y ait un spectateur ou mille. Une partie par visiteur demanderait un parc
de cartes.

Un visiteur qui arrive au milieu reçoit l'état courant et reprend le fil.

**Le navigateur ne calcule rien.** Il affiche ce que le serveur lui envoie, et
c'est le moteur Python qui fait autorité (§5.2) — y compris pour les cases où
la pièce atterrit et pour la séquence de touches reconstituée.

**Jamais en même temps qu'un entraînement.** 1 Go d'inférence plus 2,3 Go
d'entraînement ne tiennent pas dans les 3,68 Gio de la carte, et un dépassement
tuerait un run de plusieurs heures. Le verrou est dans `verrou_gpu`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

# Importés au niveau du module, et non dans `construire`. Avec
# `from __future__ import annotations`, toute annotation devient une chaîne :
# FastAPI doit résoudre « WebSocket » dans les globales du module pour
# reconnaître le paramètre. Importé dans la fonction, le nom n'y figure pas, la
# route cesse d'être traitée comme un WebSocket, et la poignée de main est
# refusée par un 403 que rien n'explique.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from ..enregistrement.capture import (
    FANTOMES,
    _cellules_posees,
    _grille_en_lignes,
    _probabilites,
    _trajectoire,
)
from ..evaluation.parties import FINAL
from ..tetris.candidats import enumerer
from ..tetris.grille import GRILLE_VIDE
from ..tetris.partie import choisir
from ..tetris.sacs import sequence
from ..tetris.touches import reconstituer

GRAINE_DEPART = 3000
"""Plage dédiée au direct public (§10.5) : aucune partie diffusée ne réutilise
une graine d'entraînement, de test ou d'évaluation. Chaque partie est
journalisée avec la sienne, donc rejouable à l'identique."""

CADENCE = 1.2
"""Secondes entre deux pièces, au minimum. Le calcul en prend 0,20 ; le reste
est du temps de lecture — une pièce toutes les cinq images serait illisible."""

PAS_TOUCHE = 0.23
"""Durée d'un appui dans la page, en secondes — la même que `PAS` côté
navigateur. Une séquence longue, deux rotations et quatre déplacements, prend
donc près de deux secondes : avec la seule cadence, la pièce suivante arrivait
avant que la patte ait fini d'appuyer, et l'on ne voyait jamais la rotation."""


def attente_apres(coup: dict) -> float:
    """Le temps de laisser la page montrer tous les appuis, puis la pose."""
    return max(CADENCE, PAS_TOUCHE * len(coup["touches"]) + 0.6)


class Direct:
    """La partie en cours et la liste de ceux qui la regardent."""

    def __init__(self, noteur, plafond: int = FINAL.plafond) -> None:
        self.noteur = noteur
        self.plafond = plafond
        self.graine = GRAINE_DEPART
        self.spectateurs: set = set()
        self.dernier: dict | None = None
        self.historique: list[dict] = []

    async def diffuser(self, message: dict) -> None:
        mort = []
        texte = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        for connexion in list(self.spectateurs):
            try:
                await connexion.send_text(texte)
            except Exception:
                mort.append(connexion)
        for connexion in mort:
            self.spectateurs.discard(connexion)

    async def jouer_sans_fin(self) -> None:
        """Enchaîne les parties, indéfiniment, une pièce à la cadence fixée."""
        while True:
            await self._une_partie(self.graine)
            self.graine += 1

    async def _une_partie(self, graine: int) -> None:
        grille = GRILLE_VIDE
        flux = sequence(graine)
        from .. import graines as familles

        departage = familles.generateur_departage(graine)
        lignes = posees = 0
        self.historique = []

        await self.diffuser({"type": "debut", "graine": graine})
        while posees < self.plafond:
            lettre = next(flux)
            candidats = enumerer(grille, lettre)
            if not candidats:
                break

            notes = self.noteur.noter(candidats)
            indice = choisir(notes, departage)
            probabilites = _probabilites(notes)
            meilleurs = sorted(range(len(notes)), key=lambda i: -notes[i])[:FANTOMES]
            choisi = candidats[indice]

            coup = {
                "type": "coup",
                "grille": _grille_en_lignes(grille),
                "piece": lettre,
                "candidats": len(candidats),
                "choix": {
                    "rotation": choisi.rotation,
                    "colonne": choisi.colonne,
                    "cellules": _cellules_posees(grille, lettre, choisi),
                },
                "lignes": choisi.lignes_completees,
                "touches": [t.value for t in reconstituer(lettre, choisi.rotation, choisi.colonne)],
                "trajectoire": _trajectoire(lettre, choisi.rotation, choisi.colonne),
                "fantomes": [
                    {
                        "rotation": candidats[i].rotation,
                        "colonne": candidats[i].colonne,
                        "probabilite": round(float(probabilites[i]), 4),
                        "cellules": _cellules_posees(grille, lettre, candidats[i]),
                    }
                    for i in meilleurs
                ],
                "total_lignes": lignes + choisi.lignes_completees,
                "numero": posees + 1,
                "graine": graine,
            }
            self.dernier = coup
            self.historique.append(coup)
            await self.diffuser(coup)

            grille = choisi.grille
            lignes += choisi.lignes_completees
            posees += 1
            await asyncio.sleep(attente_apres(coup))

        await self.diffuser({"type": "fin", "lignes": lignes, "pieces": posees, "graine": graine})
        await asyncio.sleep(2.5)


def construire(noteur, racine_web: Path, nom: str = "mouche"):
    """Rend une application FastAPI qui sert la page et diffuse la partie."""
    direct = Direct(noteur)

    @contextlib.asynccontextmanager
    async def cycle(_app):
        tache = asyncio.create_task(direct.jouer_sans_fin())
        yield
        tache.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tache

    app = FastAPI(lifespan=cycle)

    @app.websocket("/direct")
    async def flux_direct(connexion: WebSocket) -> None:
        await connexion.accept()
        direct.spectateurs.add(connexion)
        try:
            # Un arrivant reçoit d'abord de quoi rattraper la partie en cours,
            # sinon il attend la cadence entière devant une grille vide.
            await connexion.send_text(json.dumps({
                "type": "bonjour", "nom": nom, "cadence": CADENCE,
                "rattrapage": direct.historique[-1:],
            }, ensure_ascii=False))
            while True:
                await connexion.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            direct.spectateurs.discard(connexion)

    app.mount("/", StaticFiles(directory=str(racine_web), html=True), name="web")
    return app
