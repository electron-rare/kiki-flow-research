# QueryConditionedF — options d'expansion

**Version** : 2026-04-22 (révisée)
**Statut** : preprint arXiv en ligne de mire, pas de workshop peer-review ciblé par défaut.

---

## Principe

Le draft actuel fait ~3 pages (212 l. LaTeX). Deux chemins possibles
selon ce que tu veux en faire :

- **Chemin léger** — déposer tel quel sur arXiv comme short preprint,
  avec Zenodo DOI sur le tag correspondant. Suffit à créer un proof
  point citable. Rien à faire d'autre que builder et déposer.
- **Chemin étendu** — swap les 6 sections v2 en place pour atteindre
  ~8 pages (format typique workshop si un jour un workshop pertinent
  apparaît, et asset plus polish pour montrer à sponsors/financeurs).

Les expansions v2 sont livrées mais **optionnelles**. Ne les applique
que si tu veux un paper plus étoffé, pas par obligation.

---

## Expansions v2 livrées (2026-04-22)

Les 6 sections en version expanded, prêtes à swap :

| Section | Fichier v2 | Avant → Après |
|---------|-----------|--------------:|
| §1 Introduction | `section_1_intro_v2.tex` | 10 l. → ~70 l. |
| §2 Related Work | `section_2_related_v2.tex` | 14 l. → ~70 l. |
| §3 Method Overview | `section_3_method_overview_v2.tex` | 6 l. → ~50 l. |
| §4 Method | `section_4_method_v2.tex` | 63 l. → ~110 l. |
| §5 Experiments | `section_5_experiments_v2.tex` | 62 l. → ~130 l. |
| §6 Discussion | `section_6_discussion_v2.tex` | 11 l. → ~75 l. |

Total après swap : ~3 pages → ~8 pages.

**Pour basculer** (si tu veux le chemin étendu) :

```bash
cd paper/workshop
for s in 1_intro 2_related 3_method_overview 4_method 5_experiments 6_discussion; do
  mv "section_${s}.tex" "section_${s}_v1.tex"
  mv "section_${s}_v2.tex" "section_${s}.tex"
done
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Citations à ajouter dans `references.bib`

Les drafts v2 utilisent ces clés BibTeX qui ne sont peut-être pas
encore dans `references.bib`. À vérifier et ajouter si tu fais le
swap :

```bibtex
@article{chen2018neural,
  author  = {Chen, Ricky T. Q. and Rubanova, Yulia
             and Bettencourt, Jesse and Duvenaud, David},
  title   = {Neural Ordinary Differential Equations},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2018},
  volume  = {31}
}

@inproceedings{rezende2015variational,
  author    = {Rezende, Danilo Jimenez and Mohamed, Shakir},
  title     = {Variational Inference with Normalizing Flows},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2015},
  pages     = {1530--1538}
}

@inproceedings{reimers2019sbert,
  author    = {Reimers, Nils and Gurevych, Iryna},
  title     = {{Sentence-BERT}: Sentence Embeddings using Siamese
               {BERT}-Networks},
  booktitle = {Conference on Empirical Methods in Natural Language
               Processing (EMNLP)},
  year      = {2019}
}

@inproceedings{vandenoord2017neural,
  author    = {van den Oord, A{\"a}ron and Vinyals, Oriol
               and Kavukcuoglu, Koray},
  title     = {Neural Discrete Representation Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}

@article{baddeley1974,
  author  = {Baddeley, Alan D. and Hitch, Graham},
  title   = {Working Memory},
  journal = {Psychology of Learning and Motivation},
  year    = {1974},
  volume  = {8},
  pages   = {47--89}
}

@misc{saillant2026nervewml,
  author       = {Saillant, Cl{\'e}ment},
  title        = {{nerve-wml}: A Substrate-Agnostic Nerve Protocol
                  for Inter-Module Communication in Hybrid Neural Systems},
  year         = {2026},
  howpublished = {arXiv preprint},
  note         = {arXiv-id TBD --- \url{https://github.com/hypneum-lab/nerve-wml}}
}
```

---

## Dépôt arXiv

Si / quand tu déposes :

- Catégorie primaire : `cs.LG`. Cross-list possible : `cs.CL`, `q-bio.NC`.
- Abstract : <1500 chars (le draft actuel fait ~850 chars, ok).
- Comments field : mentionner le repo + DOI Zenodo + (si relevant)
  le companion paper 1 arXiv-id.
- Licence : CC BY 4.0.
