"""Diffusion du direct public et mode duel (conception.md §13.3, §13.4).

- `diffusion`  — **une seule partie**, jouée en continu, envoyée à tous les
                 navigateurs connectés. C'est la décision qui rend le direct
                 public possible : le coût GPU est constant, 0,3 s par pièce,
                 qu'il y ait un spectateur ou mille. Une partie par visiteur
                 demanderait un parc de cartes
- `duel`       — une partie par joueur, **en local uniquement**
- `verrou_gpu` — exclusion mutuelle entre entraînement et diffusion

Le verrou n'est pas un détail : 1 Go d'inférence plus 2,3 Go d'entraînement font
3,3 Go, et la carte en a 4 moins l'affichage du bureau. Ça passerait peut-être,
et un dépassement de mémoire tuerait un run de quatre heures. Donc :
l'entraînement est prioritaire, lancer un run coupe la diffusion, et le serveur
reprend la main tout seul à la fin du run.

Le direct public est **en lecture seule** : aucun visiteur ne peut agir sur la
partie ni déclencher un calcul. Le seul mode qui accepte des coups est le duel.

La machine n'est pas ouverte sur internet : tunnel sortant, sans port entrant ni
adresse fixe, qui sert aussi de coupe-circuit.

Dégradation quand ça monte : au-delà de 50 spectateurs le nuage passe à 1 000
neurones, au-delà de 200 tout le monde bascule sur le mode secours, puis on
ferme le tunnel.
"""
