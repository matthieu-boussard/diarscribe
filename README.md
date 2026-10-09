# diarscribe

Transcription diarisée, par lots :

- **Qui parle quand** : [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization) (Sortformer, 8 locuteurs max)
- **Ce qui est dit** : [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) (2B, 14 langues dont le français, Apache 2.0) par défaut, ou [`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF) (7B, accepte des mots-clés) avec `--asr vibevoice`
- **Quand chaque mot est dit** : alignement forcé CTC avec [`jonatasgrosman/wav2vec2-large-xlsr-53-french`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french) (l'aligneur de WhisperX pour le français)
- **Garde anti-boucle** : les décodeurs autorégressifs peuvent répéter la même phrase en boucle ; voir plus bas.

Fonctionne sur GPU NVIDIA (`cuda`), Apple Silicon (`mps`) ou CPU.

Mesuré sur un MacBook M4 Pro (24 Go), réunion de 45 min en français : **82 s** au total
(diarisation 9 s, transcription de 188 fenêtres 44 s, alignement 30 s), ~2,3 Go de RAM.

## Installation

```bash
# macOS : brew install ffmpeg — Linux : apt install ffmpeg
uv venv --python 3.12 && uv pip install -e ".[dev]"
hf auth login
```

Cohere Transcribe est soumis à acceptation : acceptez les conditions sur la
[page du modèle](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) avec le compte utilisé par `hf auth login`.

## Utilisation

```bash
diarscribe reunion.m4a -v                        # français par défaut
diarscribe *.wav --lang en -o transcripts/ --batch-size 32
diarscribe reunion.m4a --asr vibevoice --context "boussard, craft ai, cabinet Liins, gestion de patrimoine"
```

Sorties : `reunion.txt`, `reunion.srt`, `reunion.json` à côté du fichier (ou dans `-o DIR`).

```
[00:00:00.48 - 00:00:07.12] SPEAKER_0: Bonjour à tous, on commence ?
[00:00:07.40 - 00:00:09.05] SPEAKER_1: Oui, allons-y.
```

Fonctionnement (mode `window`, par défaut) :

1. Nemotron trouve qui parle quand.
2. Les tours sont regroupés en **fenêtres de parole continues** (≤ 30 s, coupées sur les silences > 1,5 s), et Cohere Transcribe les transcrit **par lots**. Le modèle voit des phrases entières, plus des fragments de ~1 s : c'est le principal gain de qualité.
3. wav2vec2 aligne chaque mot sur l'audio (Viterbi CTC, `align.py`).
4. Chaque mot prend le locuteur Nemotron actif à cet instant, puis un vote majoritaire par phrase (ponctuation de Cohere) corrige les mots de bord mal placés. Une bascule d'au moins 4 mots sans ponctuation est conservée.

Le mode `--mode turns` (un appel par tour de parole) reste disponible : attribution stricte, mais beaucoup moins de contexte.

### Choisir le moteur de transcription

Les deux moteurs partagent tout le reste du pipeline : fenêtres, garde anti-boucle, alignement et attribution des locuteurs.
Le JSON de VibeVoice (avec ses propres locuteurs et horodatages) est réduit à du texte.

| | `--asr cohere` (défaut) | `--asr vibevoice` |
|---|---|---|
| Taille / RAM | 2B, ~4 Go | 7B, ~17 Go en bf16 |
| Vitesse (5 min, M4 Pro) | 8 s | ~240 s |
| Mots-clés (`--context`) | non | oui : corrige les noms propres de la liste |
| Langue | imposée (`--lang`) | détectée automatiquement |
| Texte courant | plus juste, sans « euh » | garde les hésitations, hallucine davantage |
| Lots par défaut | 16 | 1 (24 Go de Mac) ; augmenter sur GPU à grande mémoire |

Sur un extrait de réunion en français, `--context "craft ai, cabinet Liins, ..."` a fait passer VibeVoice
de « Crafty Ray » et « Cabine Evans » à « Craft ai » et « cabinet Liins ». En revanche, les expressions courantes
de la liste n'ont pas été mieux reconnues.

- Dans une fenêtre, le modèle omet parfois une courte interjection prononcée par-dessus l'autre locuteur (« ok », « d'accord »).
- `--lang` est obligatoire pour le modèle (pas de détection automatique ; `fr` par défaut). Il sert aussi à écarter les sorties dans une autre écriture.
- En mode `turns`, la parole superposée est transcrite dans le tour de chacun.
- Diarisation des fichiers > 10 min : streaming par morceaux d'environ 27 s (au lieu des 0,72 s du mode « low_latency »), soit 18× plus rapide et plus proche du résultat offline.
- Les tours de moins de 0,15 s sont ignorés. Un « Merci. » isolé sur un tour de moins d'une seconde est écarté : le décodeur en invente sur les clics et les respirations.
- Les balises `[Silence]`, `[Noise]`… et les phrases-pièges connues (« Sous-titres réalisés par… ») sont retirées (`--keep-tags` pour garder les balises).

## Comment les boucles sont empêchées

| Couche | Où | Ce qu'elle fait |
|---|---|---|
| 1. Découpage | `chunking.py`, `pipeline.py` | Le modèle ne reçoit que de la parole (jamais de longs silences, déclencheurs classiques), en tours ≤ 30 s. |
| 2. Budget de tokens | `asr.py` | `max_new_tokens = 24 + 12 × durée` : une boucle ne peut pas courir indéfiniment. |
| 3. Arrêt en direct | `antiloop.LoopStoppingCriteria` | Pour chaque ligne du lot, détecte une suite périodique (boucle exacte de tokens, ou mêmes mots répétés). Seule la ligne qui boucle s'arrête. |
| 4. Diagnostic | `antiloop.diagnose` | Taux de compression zlib > 2,4, débit de mots invraisemblable, n-gramme répété, budget épuisé. |
| 5. Relances | `asr.RETRY_LADDER` | Greedy → `repetition_penalty=1.1` → échantillonnage `T=0.3`. Seuls les tours suspects sont relancés, ensemble, par lots. Pas de `no_repeat_ngram_size`, qui interdirait les vraies répétitions orales (« non non non »). |
| 6. Découpe récursive | `Transcriber.transcribe` | Si les 3 essais bouclent, le tour est coupé en deux au point le plus calme (2 niveaux max). |
| 7. Nettoyage | `antiloop.collapse_repetitions` | En dernier recours, les répétitions sont supprimées, le texte est plafonné et le segment est marqué `⚠` / `loop_trimmed`. |

Réglages : `--max-wps` (seuil de débit), `--max-chunk`, `--batch-size`. Les seuils de détection sont dans `antiloop.py`.

## Tests

```bash
pytest -q
```

Les tests couvrent la détection de boucles (y compris par ligne dans un lot), l'alignement CTC, l'attribution par phrase, le filtrage et le découpage, sans télécharger les modèles.
