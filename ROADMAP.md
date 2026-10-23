# Roadmap

**Question.** Où l'émotion de la voix se perd-elle dans Voxtral-Mini-3B, et peut-on la récupérer ?

**Hypothèse.** Voxtral n'a été entraîné que sur des objectifs dérivés du transcript ([papier](https://arxiv.org/abs/2507.13264), §3). L'encodeur capte l'émotion, mais rien n'apprend au modèle à la dire.

## Protocole

- **Données** : CREMA-D, 7 442 clips, 91 acteurs, 12 phrases × 6 émotions. Split 70/30 par acteur, aucun acteur dans les deux.
- **Ce que le modèle sait** : une passe par clip, activations moyennées sur les positions audio (sans le padding), à la sortie de l'encodeur, de l'adaptateur et des couches 0 à 29 du décodeur. Probe linéaire par tâche (phrase, genre, émotion), 5 splits.
- **Ce que le modèle dit** : choix forcé après l'audio, par lettre et par mot, options mélangées.
- **Correction** : LoRA sur les couches 20 à 29, entraînée à dire l'émotion, testée sur une formulation jamais vue.
- **Score** : accuracy équilibrée, normalisée (0 % = hasard).

## Questions

| Question | Réponse |
| --- | --- |
| L'encodeur garde-t-il l'émotion ? | Oui, 72 % |
| L'adaptateur la filtre-t-il ? | Non, 69 % |
| Le décodeur l'efface-t-il ? | Un peu, 61 % en dernière couche |
| Le modèle la dit-il ? | Non, 0 % : c'est le probe-behavior gap |
| Pooler sur le padding change-t-il les probes ? | Un peu, −4 à −6 points en couches profondes |
| Une LoRA légère comble-t-elle l'écart ? | En partie, 37 %, sans perte sur la phrase, le genre ou la transcription |

## Pistes non faites

- Probe émotion entraînée sur 8 phrases, testée sur les 4 autres.
- Transcription hors CREMA-D (LibriSpeech) avant/après LoRA.
- Tests unitaires et CI.
