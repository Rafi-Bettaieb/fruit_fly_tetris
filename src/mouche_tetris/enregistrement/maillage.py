"""Conversion du maillage NeuroMechFly pour la scène 3D (conception.md §13.1).

Le modèle biomécanique de *Drosophila melanogaster* publié par le NeLy (EPFL) est
issu d'un **scan micro-CT d'une vraie mouche**. Il est distribué dans le paquet
`flygym` sous licence Apache-2.0 : quarante maillages STL, un par segment du
corps, simplifiés à 2 000 faces chacun, plus un modèle MuJoCo qui décrit leur
emboîtement — soixante-dix corps, chacun repéré dans celui de son parent.

**La hiérarchie est conservée, pas aplatie.** Une première version fusionnait
tout en une seule géométrie : la mouche s'affichait, mais figée. Pour qu'elle
bouge les pattes, chaque segment doit rester un objet distinct, attaché à son
parent par son articulation — faire tourner un fémur doit entraîner le tibia et
les cinq tarses qui le suivent. C'est exactement ce que décrit le XML, et il
suffit de le transmettre au lieu de l'écraser.

Le calcul est fait ici, une fois. Refaire l'assemblage en JavaScript serait une
seconde implémentation d'une géométrie qui n'a aucune raison de vivre à deux
endroits.

Le résultat n'est pas une illustration : c'est la forme mesurée d'une vraie
drosophile, au même titre que le connectome est son câblage mesuré.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SOURCE = Path("/tmp/flygym/src/flygym/assets/model/neuromechfly")
SORTIE = Path("web/lib/drosophile.bin")

_SEGMENTS = {
    "coxa": "coxa", "femur": "trochanterfemur", "tibia": "tibia",
    **{f"tarsus{i}": f"tarsus{i}" for i in range(1, 6)},
}
_CENTRAUX = {
    "thorax": "c_thorax", "head": "c_head", "rostrum": "c_rostrum",
    "haustellum": "c_haustellum", "a1a2": "c_abdomen12",
    **{f"a{i}": f"c_abdomen{i}" for i in range(3, 7)},
}
_LATERAUX = {"eye": "l_eye", "wing": "l_wing", "haltere": "l_haltere",
             "pedicel": "l_pedicel", "funiculus": "l_funiculus", "arista": "l_arista"}


def _lire_flottants(texte: str | None, defaut: list[float]) -> np.ndarray:
    if not texte:
        return np.array(defaut, dtype=np.float64)
    return np.array([float(v) for v in texte.split()], dtype=np.float64)


def _fichier_simplifie(nom_xml: str) -> tuple[str | None, bool]:
    """Rend (nom du fichier simplifié, faut-il retourner en Y).

    Le XML référence les fichiers d'origine (« RFCoxa.stl ») ; le paquet ne
    distribue que les versions simplifiées, nommées autrement et **toutes du
    côté gauche**. La droite s'obtient par symétrie, comme le fait le XML.
    """
    court = nom_xml.removeprefix("mesh_")
    bas = court.lower()
    if bas in _CENTRAUX:
        return _CENTRAUX[bas], False
    cote = court[0].upper()
    if cote in "LR":
        reste = court[1:].lower()
        if reste in _LATERAUX:
            return _LATERAUX[reste], cote == "R"
        patte = court[1].upper() if len(court) > 1 else ""
        if patte in "FMH":
            segment = _SEGMENTS.get(court[2:].lower())
            if segment:
                return f"l{patte.lower()}_{segment}", cote == "R"
    return None, False


def _lire_stl(chemin: Path) -> np.ndarray:
    """Triangles d'un STL binaire, forme (n, 3, 3).

    Les normales du fichier sont ignorées : on les recalcule, sinon une symétrie
    les laisserait pointer vers l'intérieur.
    """
    donnees = chemin.read_bytes()
    (nombre,) = struct.unpack("<I", donnees[80:84])
    brut = np.frombuffer(donnees, dtype=np.uint8, count=nombre * 50, offset=84).reshape(nombre, 50)
    return np.frombuffer(brut[:, 12:48].tobytes(), dtype="<f4").reshape(nombre, 3, 3).astype(
        np.float64
    )


def _groupe(nom_corps: str) -> int:
    """0 = corps · 1 = yeux · 2 = ailes · 3 = pattes.

    Le scan ne porte aucune couleur : elle vient de l'anatomie, pas du fichier.
    """
    bas = nom_corps.lower()
    if "eye" in bas or "retina" in bas:
        return 1
    if "wing" in bas or "haltere" in bas:
        return 2
    if any(s in bas for s in ("coxa", "femur", "tibia", "tarsus", "claw")):
        return 3
    return 0


def _souder(triangles: np.ndarray, precision: int = 6):
    """Fusionne les sommets confondus d'un segment et lisse ses normales.

    Le STL décrit chaque triangle isolément : un sommet partagé par six faces y
    figure six fois. Souder divise le volume par six et, en moyennant les
    normales, donne l'ombrage lisse qu'appelle une forme organique.
    """
    aretes1 = triangles[:, 1] - triangles[:, 0]
    aretes2 = triangles[:, 2] - triangles[:, 0]
    par_face = np.cross(aretes1, aretes2)
    longueurs = np.linalg.norm(par_face, axis=1, keepdims=True)
    par_face = par_face / np.where(longueurs == 0, 1, longueurs)

    plats = triangles.reshape(-1, 3)
    _, premiers, inverse = np.unique(
        np.round(plats, precision), axis=0, return_index=True, return_inverse=True
    )
    positions = plats[premiers]
    normales = np.zeros_like(positions)
    np.add.at(normales, inverse, np.repeat(par_face, 3, axis=0))
    longueurs = np.linalg.norm(normales, axis=1, keepdims=True)
    return positions, normales / np.where(longueurs == 0, 1, longueurs), inverse.astype(np.uint32)


def assembler(source: Path = SOURCE) -> list[dict]:
    """Rend un segment par corps du modèle, exprimé dans le repère du parent."""
    xml = next((source / "legacy").glob("*deepfly3d*.xml"))
    racine = ET.parse(xml).getroot()
    corps_racine = racine.find(".//worldbody/body")
    if corps_racine is None:
        raise RuntimeError(f"aucun corps racine dans {xml}")

    echelles = {
        m.get("name"): np.abs(_lire_flottants(m.get("scale"), [1, 1, 1]))
        for m in racine.findall(".//asset/mesh")
    }
    dossier = source / "meshes" / "simplified_max2000faces"
    segments: list[dict] = []

    def descendre(corps: ET.Element, parent: int) -> None:
        nom = corps.get("name", "")
        indice = len(segments)
        segments.append({
            "nom": nom,
            "parent": parent,
            "position": _lire_flottants(corps.get("pos"), [0, 0, 0]),
            "quaternion": _lire_flottants(corps.get("quat"), [1, 0, 0, 0]),
            "groupe": _groupe(nom),
            "positions": np.zeros((0, 3)),
            "normales": np.zeros((0, 3)),
            "indices": np.zeros(0, dtype=np.uint32),
        })
        for geom in corps.findall("geom"):
            nom_maillage = geom.get("mesh")
            if not nom_maillage:
                continue
            fichier_court, retourner = _fichier_simplifie(nom_maillage)
            fichier = dossier / f"{fichier_court}.stl" if fichier_court else None
            if fichier is None or not fichier.exists():
                continue
            echelle = echelles.get(nom_maillage, np.ones(3)).copy()
            if retourner:
                echelle[1] *= -1
            triangles = _lire_stl(fichier) * echelle
            if retourner:
                # La symétrie inverse le sens de parcours des triangles : on le
                # remet à l'endroit, sinon toutes les faces regardent dedans.
                triangles = triangles[:, ::-1, :]
            positions, normales, indices = _souder(triangles)
            segments[indice]["positions"] = positions
            segments[indice]["normales"] = normales
            segments[indice]["indices"] = indices
        for enfant in corps.findall("body"):
            descendre(enfant, indice)

    descendre(corps_racine, -1)
    return segments


def ecrire(segments: list[dict], chemin: Path = SORTIE) -> Path:
    """Écrit la hiérarchie et les géométries en un seul tampon binaire.

    Format `MOUCHE3`, petit-boutiste. En-tête : nombre de segments. Puis, par
    segment — longueur et octets du nom, indice du parent (entier 16 bits signé,
    −1 pour la racine), position et quaternion, groupe, nombre de sommets et
    d'indices. Enfin les géométries concaténées.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("wb") as fichier:
        fichier.write(b"MOUCHE3")
        fichier.write(struct.pack("<I", len(segments)))
        for s in segments:
            nom = s["nom"].encode("utf-8")[:255]
            fichier.write(struct.pack("<B", len(nom)))
            fichier.write(nom)
            fichier.write(struct.pack("<h", s["parent"]))
            fichier.write(s["position"].astype("<f4").tobytes())
            fichier.write(s["quaternion"].astype("<f4").tobytes())
            fichier.write(struct.pack("<BII", s["groupe"], len(s["positions"]), len(s["indices"])))
        for s in segments:
            fichier.write(s["positions"].astype("<f4").tobytes())
            fichier.write(s["normales"].astype("<f4").tobytes())
            fichier.write(s["indices"].astype("<u4").tobytes())
    return chemin
