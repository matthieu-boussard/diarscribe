# diarscribe

Transcription diarisée, par lots :

- **Qui parle quand** : [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization) (Sortformer, 8 locuteurs max)
- **Ce qui est dit** : [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) (2B, 14 langues dont le français, Apache 2.0) par défaut, ou au choix avec `--asr` : [`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF), [`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech), [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b), [`bofenghuang/whisper-large-v3-french`](https://huggingface.co/bofenghuang/whisper-large-v3-french)
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

Tous les moteurs partagent le reste du pipeline : fenêtres, garde anti-boucle, alignement et attribution des locuteurs.

| `--asr` | Modèle | Taille | Mots-clés (`--context`) | Langue | Fenêtre max | Lots par défaut |
|---|---|---|---|---|---|---|
| `cohere` (défaut) | Cohere Transcribe 03-2026 | 2B | non | imposée | 30 s | 16 |
| `vibevoice` | VibeVoice-ASR | 7B | oui | auto | 30 s | 1 (24 Go de Mac) |
| `whisper-fr` | Whisper large-v3 affiné en français | 1,5B | oui (`prompt_ids`) | imposée | 30 s | 16 |

Notes d'intégration :
- **Whisper** : `generate` retire le token de fin de la ligne la plus longue du lot. La détection « budget de tokens épuisé » tient donc compte de la longueur réelle.

#### Comparatif (extrait de 5 min, réunion en français, MacBook M4 Pro 24 Go)

IBM Granite Speech 4.0 1B et 4.1 2B ont été évalués puis retirés des options : voir leurs lignes ci-dessous.

Contexte : `--context "boussard, craft ai, cabinet Liins, gestion de patrimoine"`. Il n'y a pas de transcription de référence :
les termes cochés sont ceux confirmés par l'utilisateur, et « écart moyen » est la distance d'édition en mots
(hésitations exclues) avec les autres sorties. Un écart faible signifie que le modèle s'accorde avec les autres, pas qu'il a raison.

| Modèle | Transcription (s) | Mots | Relances anti-boucle | Hésitations gardées | Majuscules | cabinet Liins | Craft AI | gestion de patrimoine | Écart moyen |
|---|---|---|---|---|---|---|---|---|---|
| Cohere Transcribe | **8** | 681 | 0 | 2 | 87 % | — | — | ✅ | 51 % |
| VibeVoice-ASR | 257 | 720 | 2 | 32 | 85 % | — | — | — | 51 % |
| VibeVoice-ASR + contexte | 237 | 723 | 0 | 27 | 80 % | ✅ | ✅ | — | 52 % |
| Granite 4.0 1B | 12 | 448 | 2 | 0 | 0 % | — | — | — | 69 % |
| Granite 4.0 1B + contexte | 12 | 430 | 2 | 0 | 0 % | — | — | ✅ | 73 % |
| Granite 4.1 2B | 21 | 434 | 4 | 0 | 71 % | — | — | — | 64 % |
| Granite 4.1 2B + contexte | 21 | 469 | 5 | 0 | 0 % | ✅ | — | — | 64 % |
| Whisper large-v3 FR | 22 | 674 | **0** | 12 | 70 % | — | — | ✅ | **50 %** |
| Whisper large-v3 FR + contexte | 29 | 597 | 5 | 7 | 68 % | ✅ | — | ✅ | 55 % |

Écart deux à deux (sans contexte, VibeVoice avec) : Cohere ↔ Whisper 27 %, Cohere ↔ VibeVoice 33 %, Whisper ↔ VibeVoice 36 %,
et Granite ↔ tous les autres 65 à 72 %.

À retenir :
- **Cohere** : le plus rapide, le plus proche du consensus, peu d'hésitations ; ne prend pas de mots-clés.
- **Whisper FR** : très proche de Cohere en qualité (le plus proche du consensus), environ 3× plus lent, et accepte un contexte. Le contexte corrige les noms propres de la liste mais déclenche des boucles.
- **VibeVoice** : seul à reconnaître « Craft AI » (avec contexte), garde les hésitations ; environ 30× plus lent.
- **Granite 4.0 / 4.1** (retirés) : rapides, mais rendent ~35 % de mots en moins. Sur les échanges rapides à deux voix, ils résument ou inventent du texte (4.1 : « nommé d'après la ville de Coftea, en Moldavie… » sur un dialogue de dépannage navigateur). Ils exigeaient en plus des contournements : prompt nommant la langue (sinon 4.1 traduit en anglais), fenêtres de 12 s pour 4.0, et un correctif de `transformers` 5.19.

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
