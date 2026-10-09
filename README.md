# diarscribe

Transcription diarisée pour Mac (Apple Silicon) :

- **Qui parle quand** : [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization) (Sortformer, 8 locuteurs max)
- **Ce qui est dit** : [`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF) (multilingue, mots-clés de contexte)
- **Garde anti-boucle** : VibeVoice a tendance à répéter la même phrase en boucle ; voir plus bas.

## Installation

```bash
brew install ffmpeg
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

`hf auth login` peut être nécessaire pour télécharger Nemotron (modèle à licence OpenMDW).

Mémoire : VibeVoice-ASR pèse ~17 Go en bf16. Sur un Mac de 24 Go, fermez les applis lourdes.
Nemotron est petit et bascule tout seul sur CPU si MPS échoue.

## Utilisation

```bash
diarscribe reunion.m4a --lang fr -v
diarscribe interview.mp3 --lang fr -c "Craft AI, Matthieu, diarisation" -f txt,srt
```

Sorties : `reunion.txt`, `reunion.srt`, `reunion.json` à côté du fichier (ou dans `-o DIR`).

```
[00:00:00.48 - 00:00:07.12] SPEAKER_0: Bonjour à tous, on commence ?
[00:00:07.40 - 00:00:09.05] SPEAKER_1: Oui, allons-y.
```

Modes :
- `window` (défaut) : fenêtres de parole ≤ `--max-chunk` s (30 par défaut). Chaque phrase de VibeVoice est attribuée au locuteur Nemotron qui la recouvre le plus. Le contexte long améliore nettement la reconnaissance. Limite : une réponse très courte de l'autre locuteur (« Exactement. ») peut rester collée à la phrase voisine.
- `turns` : un appel par tour de parole Nemotron. L'attribution est stricte, mais les tours de moins d'une seconde font halluciner le modèle (chinois, « Make sure to subscribe »…).

`--lang fr` écarte les sorties dans une autre écriture (chinois, cyrillique…). Les balises `[Silence]`, `[Noise]` et `[Unintelligible Speech]`, ainsi que les phrases-pièges connues, sont retirées par défaut (`--keep-tags` pour garder les balises).

Mesuré sur un M4 Pro (24 Go), réunion en français : ~1,2× la durée de l'audio, ~18 Go de RAM.

## Comment les boucles sont empêchées

| Couche | Où | Ce qu'elle fait |
|---|---|---|
| 1. Découpage | `chunking.py`, `pipeline.py` | VibeVoice ne reçoit que de la parole (jamais de longs silences, déclencheurs classiques), en segments ≤ 30 s coupés au point le plus silencieux. |
| 2. Budget de tokens | `asr.py` | `max_new_tokens = 80 + 15 × durée` : une boucle ne peut pas courir indéfiniment. |
| 3. Arrêt en direct | `antiloop.LoopStoppingCriteria` | À chaque token, détecte une suite périodique : boucle exacte de tokens, ou même texte répété avec seulement les timestamps qui changent. `generate` s'arrête aussitôt. |
| 4. Diagnostic | `antiloop.diagnose` | Taux de compression zlib > 2,4, plus de 6 mots/s, n-gramme répété, timestamps hors audio ou qui reculent, budget épuisé. |
| 5. Relances | `asr.RETRY_LADDER` | Greedy → `repetition_penalty=1.1` → échantillonnage `T=0.3`. Pas de `no_repeat_ngram_size`, qui casserait le JSON produit par VibeVoice. |
| 6. Découpe récursive | `Transcriber.transcribe_span` | Si les 3 essais bouclent, le segment est coupé en deux au point le plus calme (2 niveaux max). |
| 7. Nettoyage | `antiloop.collapse_repetitions` | En dernier recours, les répétitions sont supprimées, le texte est plafonné et le segment est marqué `⚠` / `loop_trimmed`. |

Réglages : `--max-wps` (seuil de débit), `--max-chunk`. Les seuils de détection sont dans `antiloop.py`.

## Tests

```bash
pytest -q
```

Les tests couvrent la détection de boucles et le découpage, sans télécharger les modèles.
