"""Le cerveau : graphe mesuré, signes, modèle à taux et variantes de contrôle.

Conception.md §7, §8, §11.2. 165 122 neurones tracés, 25 563 197 connexions.

Modules prévus :

- `acquisition` — appel aux scripts de flyhard, lecture par morceaux du fichier
                  de 1,1 Go et de ses 152 millions de lignes (§4.2)
- `empreintes`  — SHA-256 vérifiée à **chaque** chargement ; empreinte différente
                  de celle de flyhard ⇒ arrêt, retour à la version publiée (§16)
- `signes`      — loi de Dale : ACh +, GABA −, glutamate −, histamine −, reste + (§7.3)
- `graphe`      — normalisation par neurone postsynaptique, facteur global,
                  mesure de D, participation, composition des interfaces (§7.4, §7.6)
- `modele`      — la dynamique à taux et les 25 728 320 paramètres appris (§8)
- `controles`   — recâblée, recâblage à signes conservés, réservoir, graphe coupé,
                  entrée aveuglée (§11.2)

Invariants que ce paquet ne doit jamais enfreindre :

- **câblage figé** : aucune connexion ajoutée, retirée ou déplacée ;
- **seuls s'entraînent** un gain par connexion (softplus, donc jamais négatif :
  un gain ne peut pas inverser un signe), une fuite par neurone (sigmoïde) et
  une température ;
- **ne jamais matérialiser le message de chaque connexion** : 25,6 M × 40 × 4
  octets ≈ 4 Go pour une seule mise à jour. Utiliser le noyau creux de flyhard
  avec le correctif CUDA de fly-self-driving ; ne pas en écrire un nouveau (§14.2) ;
- **tout le signal passe par le graphe** : populations d'entrée et de sortie
  disjointes, interfaces aléatoires et figées, aucun chemin programmé (§9.3).
"""
