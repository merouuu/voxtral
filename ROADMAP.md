# Paralinguistic Probing of Voxtral-Mini-3B — Roadmap

> **Objectif en une phrase.** Montrer, avec un pipeline PyTorch propre et une seule figure, *où* dans l'architecture de Voxtral-Mini-3B l'information prosodique (émotion) survit, et *où* elle est perdue — et relier ce comportement à la recette d'entraînement décrite dans le papier Voxtral (arXiv:2507.13264).

> **Pourquoi c'est pertinent pour l'offre Mistral (Applied Scientist, Solutions).** Modèle multimodal de Mistral + interprétabilité rigoureuse + code production-grade (typage, tests, CI). Le projet ponte directement les deux expériences du CV (audio/embeddings gelés chez Deezer, interprétabilité à Polytechnique Montréal).

---

## 0. Le message scientifique (à garder en tête tout le long)

Voxtral est entraîné, de bout en bout, avec des **objectifs purement textuels dérivés du transcript** :

| Phase (papier) | Objectif | Signal paralinguistique ? |
|---|---|---|
| Pré-entraînement §3.1 | `<repeat>` : audio → transcript ; `<next>` : audio → texte suivant | Aucun |
| Warm-up adaptateur §3.1 | Encodeur + décodeur gelés, seul l'adaptateur apprend, sur ces mêmes patterns | Aucun |
| SFT §3.2 | QA générées par Mistral Large **à partir du transcript** ; audio-only via TTS | Aucun (le générateur ne voit jamais l'audio) |
| DPO / Online DPO §3.3 | Reward model **texte-only** qui lit la transcription à la place de l'audio | Aucun |
| Évaluations §3.4 | GSM8K / TriviaQA / MMLU synthétisés en parole + SU benchmark jugé sur transcript | Aucune mesure |

**Hypothèse centrale (H0).** Comme aucun objectif n'exige la prosodie, le modèle n'a aucune raison de la *transmettre* jusqu'à la génération, même si l'encodeur Whisper la *capture*. On s'attend à : émotion décodable linéairement en sortie d'encodeur → atténuée dans le décodeur → quasi-nulle au niveau comportemental. Le genre (attribut acoustique bas-niveau) et le contenu lexical (ce pour quoi le modèle est entraîné) servent de contrôles.

Quel que soit le résultat, le projet est publiable sur GitHub : voir §7 « Scénarios ».

---

## 1. Questions de recherche, ancrées dans le papier

Chaque question cite la section exacte du papier, formule une hypothèse falsifiable et dit **quelle mesure du pipeline y répond**. C'est ce tableau qui structure le README final.

| # | Question précise | Ancrage papier | Hypothèse | Mesure |
|---|---|---|---|---|
| **Q1** | L'encodeur Whisper large-v3 de Voxtral (gelé pendant le warm-up, puis fine-tuné) conserve-t-il l'émotion linéairement décodable, comme Whisper standard ? | §2.1 : encodeur = Whisper large-v3, 50 Hz, padding à 30 s | Oui : probe émotion ≫ hasard (norm. > 50 %) | Probe sur `encoder_out` (1280-d, mean-pool masqué) |
| **Q2** | L'adaptateur MLP (×4 downsampling → 12,5 Hz), entraîné *seul* d'abord avec encodeur et décodeur gelés sur des objectifs lexicaux, agit-il comme un **filtre** qui jette la prosodie ? | §2.2 (adaptateur) ; §3.1 « we freeze the audio encoder and language decoder, training only the adapter » | L'adaptateur préserve l'émotion (compression temporelle, pas sémantique) : chute < 10 pts entre encodeur et adaptateur | Probe sur `adapter_out` (3072-d) vs `encoder_out` |
| **Q3** | Le papier conjecture qu'à 12,5 Hz « each audio-embedding encodes a similar amount of information as a text-embedding ». Si c'est vrai, les tokens audio sont-ils déjà « text-like », i.e. le genre et l'émotion s'effacent-ils dès l'entrée du décodeur ? | §5.2 (hypothèse 12,5 Hz) | Non : genre et émotion restent présents à la couche 0 ; le « text-likeness » apparaît progressivement | Probes couche 0 vs couches 8–16 |
| **Q4** | Dans quelle(s) couche(s) du décodeur l'émotion disparaît-elle, et le genre survit-il plus longtemps ? | §2.3 : décodeur Ministral 3B | Émotion : décroissance monotone sur les couches ; genre : plateau ; texte : plateau ~97 % | Trajectoire des 3 probes sur 8 couches |
| **Q5** | Le SFT génère ses QA depuis le transcript et demande explicitement à Mistral Large de les « framer comme de la compréhension auditive ». Le modèle a-t-il donc appris à *simuler* l'écoute sans utiliser la prosodie ? | §3.2 « Audio Context, Text Query » | Accuracy comportementale émotion ≈ hasard, alors que la même info est décodable linéairement dans ses activations (**modality gap**) | Choix forcé par log-vraisemblance vs probe |
| **Q6** | Le reward model DPO ne voit que la transcription (« captures semantics, style, factual coherence »). Peut-on observer une trace de ce biais : le modèle est-il *confiant* dans ses réponses émotionnelles erronées ? | §3.3 | Distribution des log-probs sur les 6 émotions peu entropique et dominée par `neutral`/`happiness` | Entropie + matrice de confusion du Bloc comportemental |
| **Q7** | Les trois benchmarks « speech understanding » publiés sont des benchmarks **texte** synthétisés en parole. Que mesurerait un benchmark qui exige l'audio brut ? | §3.4 | CREMA-D en choix forcé est exactement ce benchmark : lexique constant, seul le canal prosodique varie | Toute la figure |
| **Q8** (contrôle) | Le padding Whisper à 30 s (§2.1, §5.1) dilue-t-il les représentations si l'on pool naïvement ? | §2.1, §5.1 | Oui : pooler sur tous les tokens vs tokens non-paddés change fortement la probe | Ablation masque de pooling (1 ligne de code) |

Q1–Q5 + Q8 sont obligatoires. Q6 et Q7 sont de la discussion dans le README, quasi gratuites.

---

## 2. Voxtral-Mini-3B : ce qu'on sonde, avec les dimensions

À vérifier contre `config.json` du modèle HF et `print(model)` avant de coder les hooks.

```
waveform (16 kHz)
  └─ log-Mel 128 bins, hop 160                               (§2.1)
  └─ Whisper large-v3 encoder (640M)  → [T_enc, 1280], 50 Hz   (§2.1, Table 1)   ← HOOK A  "encoder"
  └─ Adapter MLP (25M): stack 4 frames → [T_enc/4, 3072], 12.5 Hz (§2.2)        ← HOOK B  "adapter"
  └─ Décodeur Ministral-3B (3.6B), ~30 couches, hidden 3072    (§2.3, Table 1)
        résidu couche ℓ, positions des tokens audio uniquement                   ← HOOKS C  ℓ ∈ {0,4,8,12,16,20,24,29}
  └─ logits du prochain token                                                   ← Bloc comportemental
```

Points techniques à régler dès le jour 0 :

- **Positions audio** : dans HF, `input_ids == model.config.audio_token_id` donne le masque des positions audio dans la séquence du décodeur.
- **Masque de padding (Q8)** : un clip CREMA-D dure ~2–3 s mais Whisper pad à 30 s → 1500 frames encodeur → 375 tokens audio. Ne pooler que les `ceil(durée_s × 12.5)` premiers tokens (resp. `ceil(durée_s × 50)` frames encodeur). Garder la version « tous tokens » comme ablation.
- **Prompt fixe neutre** pour l'extraction (ex. `"Describe this audio."`), identique pour tous les clips, pour que la seule variable soit l'audio.
- **Précision** : bf16 sur A100/L4 ; fp16 sur T4. Les activations sont sauvées en fp16.

---

## 3. Protocole minimal

### 3.1 Dataset : CREMA-D, et rien d'autre

- 7 442 clips, 91 acteurs (48 H / 43 F), **12 phrases fixes**, **6 émotions** (anger, disgust, fear, happiness, neutral, sadness).
- Pourquoi c'est le dataset idéal pour Q5/Q7 : chaque phrase est dite dans chaque émotion → **l'émotion est orthogonale au lexique par construction**. Un probe ne peut pas tricher avec le contenu.
- Repères pour calibrer les résultats : hasard 6 classes = 16,7 % ; humains audio-only ≈ 41 % (Cao et al., 2014) ; modèles SER dédiés (WavLM/wav2vec2 fine-tunés) ≈ 70–80 %.
- Source : dépôt original `CheyneyComputerScience/CREMA-D` ou un miroir HuggingFace (vérifier la licence ODbL et citer).

### 3.2 Split

- **Strictement par acteur** : 70 % des acteurs en train, 30 % en test, seed fixée. Aucun acteur dans les deux. Sinon le probe genre/émotion mémorise des voix.
- Probing : `GroupKFold(n_splits=5)` groupé par acteur sur la partie train pour choisir `C`, puis évaluation sur les acteurs test. Rapporter moyenne ± écart-type sur 5 seeds de split.
- Le Bloc comportemental tourne sur **le même split test**, pour que les deux courbes soient comparables.

### 3.3 Trois attributs, mêmes clips, mêmes features

| Attribut | Classes | Rôle | Attendu (probe) | Attendu (comportement) |
|---|---|---|---|---|
| Texte (quelle phrase ?) | 12 | Contrôle positif lexical | ~97 % partout | Élevé (c'est de l'ASR) |
| Genre | 2 | Attribut acoustique bas-niveau | Élevé, plateau | Moyen à élevé |
| **Émotion** | 6 | Cible paralinguistique | Haut à l'encodeur, décroît | **≈ hasard** |

### 3.4 Un seul modèle

`mistralai/Voxtral-Mini-3B-2507` (checkpoint public = Online DPO, §5.4). Pas de Voxtral Small, pas de variante Transcribe, pas de projection aléatoire. Le contrôle interne est la comparaison entre les trois attributs à travers les couches.

---

## 4. Structure du dépôt

```
voxtral-paralinguistic-probing/
├── README.md                 # figure maîtresse en haut, puis Q1–Q8 et réponses
├── ROADMAP.md                # ce fichier (peut être supprimé à la fin)
├── pyproject.toml            # deps + ruff + mypy + pytest config
├── .github/workflows/ci.yml # ruff + mypy --strict + pytest (CPU, sans modèle)
├── src/vpp/
│   ├── __init__.py
│   ├── data.py               # chargement CREMA-D, parsing des noms de fichiers, split par acteur
│   ├── model.py              # chargement Voxtral, pose des hooks, masque des tokens audio
│   ├── pooling.py            # mean-pool masqué (padding 30 s)
│   ├── extract.py            # CLI 1 : 1 forward/clip → activations .npy / .safetensors
│   ├── probe.py              # CLI 2 : régression logistique L2, GroupKFold acteur
│   ├── behavioral.py         # CLI 3 : choix forcé par log-vraisemblance
│   └── plot.py               # CLI 4 : figure maîtresse
├── tests/
│   ├── test_data.py          # parsing noms CREMA-D, split sans fuite d'acteur
│   ├── test_pooling.py       # masque de padding correct sur tenseurs synthétiques
│   ├── test_probe.py         # probe sur données synthétiques séparables → 100 %
│   └── test_behavioral.py    # scoring log-vraisemblance sur un petit modèle factice
├── configs/default.yaml      # couches sondées, prompt, seed, chemins
└── outputs/                  # .gitignore ; activations, csv de résultats, figure
```

Règles :
- Tout est typé (`mypy --strict` passe). Dataclasses pour les configs. Pas de notebook dans `src/`.
- Les tests ne téléchargent **jamais** le modèle : ils testent la logique (split, masque, scoring) sur des tenseurs synthétiques, donc tournent en CI CPU en < 1 min.
- Un `Makefile` ou des entrées `[project.scripts]` : `vpp-extract`, `vpp-probe`, `vpp-behavioral`, `vpp-plot`.

---

## 5. Les quatre scripts en détail

### 5.1 `extract.py` — une seule passe sur les 7 442 clips

1. Charger `VoxtralForConditionalGeneration` + `AutoProcessor` (transformers récent + `mistral-common[audio]`).
2. Construire la conversation `[audio, "Describe this audio."]` via `processor.apply_chat_template`.
3. Hooks forward sur : sortie de l'encodeur, sortie de l'adaptateur, sortie des couches `{0,4,8,12,16,20,24,29}` du décodeur.
4. Pour chaque hook : sélectionner les positions audio, appliquer le masque de padding, mean-pool → un vecteur par clip.
5. Sauver `features[hook] : [N, d]` en fp16 + `meta.csv` (clip, acteur, phrase, émotion, genre, durée).

Budget : ~7 400 forwards de ~400 tokens sur un 3B → **< 30 min sur A100/L4**, ~1 h sur T4. Stockage ≈ 7 400 × (1280 + 3072 + 8×3072) × 2 octets ≈ **430 Mo**.

### 5.2 `probe.py` — régression logistique, pas plus

- `StandardScaler` + `LogisticRegression(penalty="l2", C ∈ {0.01, 0.1, 1, 10})`, `max_iter` suffisant.
- Métrique : **balanced accuracy**, puis normalisée `(acc − chance) / (1 − chance)` pour mettre les 3 tâches sur le même axe (0 % = hasard, 100 % = parfait).
- Sortie : `probe_results.csv` avec colonnes `hook, task, acc, acc_norm, std`.
- Contrôle bonus (10 lignes) : probe émotion entraîné sur phrases 1–8, testé sur phrases 9–12 → prouve que le probe ne s'appuie pas sur le lexique.

### 5.3 `behavioral.py` — choix forcé par log-vraisemblance

- Prompt par tâche, options lettrées pour obtenir **un seul token de réponse** → un forward par (clip, tâche) :
  - Émotion : `"What emotion does the speaker express? A) anger B) disgust C) fear D) happiness E) neutral F) sadness. Answer with the letter."`
  - Genre : `"Is the speaker male or female? A) male B) female. Answer with the letter."`
  - Texte : `"Which sentence was spoken? A) ... L) ... Answer with the letter."`
- Score = argmax des logits restreints aux lettres valides. Sauver aussi la distribution complète (pour Q6 : entropie, confusion).
- Sanity check sur 200 clips : variante « réponse en mot entier » avec log-prob sommée et normalisée par la longueur, pour vérifier que le format lettre ne biaise pas.
- Tourne sur le split test uniquement : ~2 200 clips × 3 tâches → **< 15 min sur A100**.

### 5.4 `plot.py` — LA figure

- Abscisse : `Whisper → Adapter → L0 → L4 → … → L29`.
- Ordonnée : accuracy normalisée (0–100 %).
- 3 courbes pleines avec bande ± std : Texte, Genre, Émotion (probing).
- 3 lignes horizontales pointillées, mêmes couleurs : accuracy comportementale zero-shot.
- Annotations légères : « human audio-only ≈ 41 % » sur l'émotion, et la flèche « modality gap » entre la courbe probe émotion et sa ligne pointillée.
- Export PNG 300 dpi + SVG. Placée en tout premier dans le README.

---

## 6. Planning

### J0 — Mise en place (≈ 2 h, un soir)
- [ ] Créer le repo, `pyproject.toml`, ruff/mypy/pytest configurés, CI qui passe sur un test trivial.
- [ ] Télécharger CREMA-D, écrire `data.py` + `test_data.py` (parsing `1001_DFA_ANG_XX.wav` → acteur/phrase/émotion ; genre via `VideoDemographics.csv`).
- [ ] Charger Voxtral-Mini sur GPU, **un** forward sur un clip, `print(model)`, noter les noms exacts des modules et les shapes de chaque hook. Confirmer le nombre de couches du décodeur et `audio_token_id`.

### J1 — Extraction + probing (samedi)
- [ ] `pooling.py` + `test_pooling.py` (masque padding, Q8).
- [ ] `extract.py` complet, lancer sur les 7 442 clips.
- [ ] Pendant que ça tourne : `probe.py` + `test_probe.py` sur données synthétiques.
- [ ] Lancer les probes, obtenir `probe_results.csv`. Regarder d'abord la tâche Texte : si elle n'est pas ~97 % à l'adaptateur, il y a un bug de masque ou de positions audio.

### J2 — Comportemental + figure + README (dimanche)
- [ ] `behavioral.py` + `test_behavioral.py`, lancer sur le split test.
- [ ] `plot.py`, produire la figure, l'itérer 2–3 fois jusqu'à ce qu'elle se lise en 10 secondes.
- [ ] Rédiger le README : figure, 3 phrases de résultat, tableau Q1–Q8 avec réponses chiffrées, section « How to reproduce » (4 commandes), limites.

### J3 — Buffer (une soirée)
- [ ] `mypy --strict` et `ruff` à zéro warning ; coverage des tests ≥ 80 % sur `src/`.
- [ ] Contrôle bonus cross-phrases ; ablation Q8 (une ligne dans le README).
- [ ] Mettre à jour le CV (§9) et vérifier que le lien GitHub est cliquable dans le PDF.

---

## 7. Scénarios de résultats et ce qu'on en dit

| Scénario | Lecture | Phrase clé du README |
|---|---|---|
| **A (attendu)** : émotion haute à l'encodeur, décroît dans le décodeur, comportement ≈ hasard | Modality gap : l'info existe mais n'est pas routée vers la génération | « Voxtral hears emotion but was never taught to say it » |
| **B** : émotion déjà faible à l'adaptateur | L'adaptateur, entraîné seul sur des objectifs lexicaux (§3.1), filtre la prosodie dès l'entrée | « The adapter is the bottleneck » |
| **C** : émotion stable dans tout le décodeur mais comportement ≈ hasard | Info disponible partout mais ignorée par la tête de lecture → pointe vers SFT/DPO transcript-only (§3.2–3.3) | « Linear availability ≠ behavioral access » |
| **D** : comportement bon (> 40 %) | Le modèle généralise au-delà de ses objectifs ; résultat intéressant en soi, à comparer au score humain audio-only | « Emergent prosodic sensitivity despite transcript-only supervision » |

Dans tous les cas, le genre et le texte doivent se comporter comme prévu ; sinon, c'est un bug, pas un résultat.

---

## 8. Qualité de code — mapping explicite avec l'offre

| Offre Mistral | Dans le repo |
|---|---|
| « Expert with PyTorch » | Hooks forward propres, pas de fork du modèle, batching avec masques |
| « Writes clean, readable, high-performance, fault-tolerant Python » | Typage strict, dataclasses, CLI `argparse`/`typer`, reprise d'extraction après crash (checkpoint tous les N clips) |
| « Enjoys improving existing code: typing, tests, CI pipelines » | `mypy --strict`, `ruff`, `pytest` + GitHub Actions badge dans le README |
| « Multimodal models » | Le sujet entier |
| « Excellent communication skills » | README lisible en 45 s : figure, 3 phrases, tableau Q/R |

---

## 9. Bullets CV (à coller dans PROJECTS, en remplaçant le hackathon 2024)

**Paralinguistic Probing of Voxtral-Mini-3B** | PyTorch, HuggingFace, scikit-learn — github.com/merouuu/voxtral-paralinguistic-probing

- Diagnosed a modality gap in Mistral's Voxtral-Mini-3B on 7.4k speaker-disjoint CREMA-D clips: forced-choice log-likelihood evaluation reaches >95% on lexical content but near chance on emotion, despite lexical content being held constant by design.
- Tracked where prosodic information survives via layer-wise linear probes on frozen audio-token residual streams (Whisper encoder → 12.5 Hz adapter → decoder layers), showing it is progressively attenuated under transcript-only pre-training, SFT and DPO objectives while speaker gender and semantics persist.

(Adapter les chiffres aux résultats réels. Si scénario B ou C, remplacer la deuxième puce par la phrase clé correspondante de §7.)

---

## 10. Pièges connus

- **Fuite d'acteur** : c'est le bug le plus probable et le plus embarrassant. `test_data.py` doit asserter `set(train_actors) ∩ set(test_actors) == ∅`.
- **Pooling sur le padding** : sans masque, 90 % des tokens audio sont du silence → les probes s'écroulent artificiellement. D'où Q8 comme ablation plutôt que comme erreur silencieuse.
- **Positions audio dans le décodeur** : ne pas pooler sur les tokens du prompt texte.
- **Déséquilibre léger** : `neutral` a moins de clips (une seule intensité) → balanced accuracy obligatoire.
- **Mémoire** : garder `output_hidden_states=False` et utiliser des hooks ciblés ; sinon 30 couches × 400 tokens × 3072 × batch explose.
- **Versions** : fixer `transformers`, `mistral-common`, `torch` dans `pyproject.toml` ; Voxtral a besoin d'une version récente de transformers.

---

## 11. Stretch goals (uniquement si J0–J3 sont bouclés)

1. **Attention du token de réponse** vers les tokens audio : l'attention finale regarde-t-elle les mêmes frames pour « émotion » et pour « texte » ? (une figure de plus, répond mécaniquement à Q5.)
2. **Voxtral-Small-24B** : la même courbe à 24B — le gap se réduit-il avec l'échelle ?
3. **Probe sur l'encodeur Whisper large-v3 vanilla** vs encodeur Voxtral : le fine-tuning de Voxtral a-t-il déjà érodé l'émotion dans l'encodeur ?

Ne pas commencer ces points avant que la figure principale et le README soient finis : un repo fini avec une figure vaut plus qu'un repo ambitieux à moitié fait.

---

## Références

- Voxtral — Mistral AI, arXiv:2507.13264 (juillet 2025). Sections citées : §2.1, §2.2, §2.3, §3.1, §3.2, §3.3, §3.4, §5.1, §5.2, §5.4, Table 1.
- CREMA-D — Cao et al., *IEEE Trans. Affective Computing*, 2014.
- Whisper — Radford et al., ICML 2023.
