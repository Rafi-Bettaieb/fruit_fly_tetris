"""Exclusion mutuelle entre l'entraînement et la diffusion (§13.4).

1 Go d'inférence plus 2,3 Go d'entraînement font 3,3 Gio sur une carte qui en
offre 3,68, moins ce que prend l'affichage du bureau. Cela passerait peut-être,
et un dépassement de mémoire tuerait un run de plusieurs heures.

**L'entraînement est prioritaire** : c'est lui qui produit les résultats. Lancer
un run coupe la diffusion, qui rend le verrou ; le serveur reprend la main à la
fin du run, sans intervention.

Le verrou est un simple fichier. Il porte le numéro du processus qui le détient,
ce qui permet de reconnaître un verrou abandonné — après un plantage, par
exemple — au lieu de bloquer la carte jusqu'au redémarrage.
"""

from __future__ import annotations

import os
from pathlib import Path

VERROU = Path("data/verrou-gpu")


class GpuOccupe(RuntimeError):
    """Un autre usage détient la carte."""


def _vivant(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def detenteur() -> tuple[int, str] | None:
    """Rend (pid, usage) si le verrou est tenu par un processus vivant."""
    if not VERROU.exists():
        return None
    try:
        pid_texte, usage = VERROU.read_text(encoding="utf-8").split(maxsplit=1)
        pid = int(pid_texte)
    except (ValueError, OSError):
        VERROU.unlink(missing_ok=True)
        return None
    if not _vivant(pid):
        # Verrou abandonné : le laisser bloquerait la carte jusqu'au redémarrage.
        VERROU.unlink(missing_ok=True)
        return None
    return pid, usage.strip()


def prendre(usage: str, prioritaire: bool = False) -> None:
    """Prend le verrou, ou lève `GpuOccupe`.

    `prioritaire` est réservé à l'entraînement : il évince la diffusion, qui est
    faite pour être interrompue, mais jamais un autre entraînement.
    """
    courant = detenteur()
    if courant:
        pid, tenu_par = courant
        if not (prioritaire and tenu_par == "diffusion"):
            raise GpuOccupe(
                f"la carte est prise par « {tenu_par} » (processus {pid}). "
                + ("Attendre la fin du run." if tenu_par == "entrainement"
                   else "Arrêter la diffusion, ou relancer avec la priorité.")
            )
    VERROU.parent.mkdir(parents=True, exist_ok=True)
    VERROU.write_text(f"{os.getpid()} {usage}\n", encoding="utf-8")


def rendre() -> None:
    courant = detenteur()
    if courant and courant[0] == os.getpid():
        VERROU.unlink(missing_ok=True)
