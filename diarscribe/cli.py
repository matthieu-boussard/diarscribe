"""Command line entry point: ``diarscribe audio.m4a``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ops not yet implemented on Metal fall back to CPU instead of crashing. Must be set before torch import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from . import output, pipeline  # noqa: E402
from .align import Aligner  # noqa: E402
from .asr import BACKENDS, CONTEXT_BACKENDS  # noqa: E402
from .diarize import Diarizer  # noqa: E402
from .postprocess import PostConfig, postprocess, summary  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
LANGUAGES = ["ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="diarscribe", description="Transcription diarisée (Nemotron-3-Diarization + Cohere / VibeVoice / Whisper)")
    ap.add_argument("audio", nargs="+", help="fichier(s) audio/vidéo (tout format lu par ffmpeg)")
    ap.add_argument("-o", "--output-dir", type=Path, help="dossier de sortie (défaut : à côté de l'audio)")
    ap.add_argument("-f", "--formats", default="txt,srt,json", help="formats de sortie parmi txt,srt,json")
    ap.add_argument("--asr", choices=list(BACKENDS), default="cohere",
                    help="cohere (2B, défaut) ; vibevoice (7B) ; whisper-fr (large-v3 affiné en français). "
                         "vibevoice et whisper-fr acceptent --context")
    ap.add_argument("-l", "--lang", choices=LANGUAGES, default="fr",
                    help="langue de l'audio (imposée à Cohere ; choix de l'aligneur) ; écarte aussi les sorties dans une autre écriture")
    ap.add_argument("-c", "--context", help="mots-clés / contexte (noms propres, jargon...) ; pas avec cohere")
    post = ap.add_argument_group("post-traitement du texte")
    post.add_argument("-g", "--glossary",
                      help="noms propres, séparés par des virgules ; variantes connues avec = : "
                           "\"cabinet Liins, Craft AI=crafty ray|clarge tri\" (défaut : termes de --context)")
    post.add_argument("--style", choices=["verbatim", "lu"], default="verbatim",
                      help="verbatim : tel que dit ; lu : sans hésitations (euh, hum) ni bégaiements (je je)")
    post.add_argument("--numbers", choices=["keep", "digits"], default="keep", help="digits : « dix ans » -> « 10 ans »")
    post.add_argument("--no-typography", action="store_true", help="ne pas appliquer la typographie française")
    post.add_argument("--no-lang-filter", action="store_true",
                      help="garder les segments détectés dans une autre langue (hallucinations sur fenêtres courtes)")
    ap.add_argument("--mode", choices=["window", "turns"], default="window",
                    help="window : fenêtres continues ≤ 30 s, mots alignés puis attribués au locuteur (plus de contexte) ; "
                         "turns : un appel par tour de parole")
    ap.add_argument("--align-model", help="modèle CTC wav2vec2 pour l'alignement des mots (défaut selon --lang)")
    ap.add_argument("--no-punctuation", action="store_true", help="transcription sans ponctuation ni majuscules")
    ap.add_argument("--keep-tags", action="store_true", help="garder les balises [Silence], [Noise]...")
    ap.add_argument("--batch-size", type=int,
                    help="fenêtres transcrites par appel (défaut : 16 cohere/whisper, 1 vibevoice ; augmenter sur GPU à grande mémoire)")
    ap.add_argument("--max-chunk", type=float, default=30.0, help="durée max d'un tour envoyé au modèle (s, ≤ 35)")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--diar-device", default=None, help="device pour la diarisation (défaut : --device)")
    ap.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    ap.add_argument("--max-wps", type=float, default=6.0, help="débit max plausible (mots/s) avant suspicion de boucle")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("diarscribe").setLevel(logging.INFO)
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    if unknown := set(formats) - output.WRITERS.keys():
        ap.error(f"format(s) inconnu(s) : {', '.join(sorted(unknown))}")
    if not 0 < args.max_chunk <= 35:
        ap.error("--max-chunk doit être dans ]0, 35] (au-delà, Cohere redécoupe lui-même l'audio)")
    if args.context and args.asr not in CONTEXT_BACKENDS:
        ap.error(f"--context n'est pas pris en charge par --asr {args.asr}")

    print("Chargement de Nemotron-3-Diarization...", file=sys.stderr)
    diarizer = Diarizer(device=args.diar_device or args.device)
    common = dict(device=args.device, dtype=DTYPES[args.dtype], language=args.lang, max_wps=args.max_wps)
    if args.batch_size:
        common["batch_size"] = args.batch_size
    if args.asr in CONTEXT_BACKENDS:
        common["context"] = args.context
    if args.asr == "cohere":
        common["punctuation"] = not args.no_punctuation
    print(f"Chargement du modèle de transcription ({args.asr})...", file=sys.stderr)
    transcriber = BACKENDS[args.asr](**common)
    aligner = None
    if args.mode == "window":
        print("Chargement de l'aligneur wav2vec2...", file=sys.stderr)
        aligner = Aligner(language=args.lang, device=args.device, model_id=args.align_model)

    glossary = args.glossary if args.glossary is not None else args.context
    post_cfg = PostConfig(
        lang=args.lang, lang_filter=not args.no_lang_filter, style=args.style, numbers=args.numbers,
        typography=not args.no_typography, glossary=[t for t in (glossary or "").split(",") if t.strip()],
    )

    for audio in args.audio:
        src = Path(audio)
        segments = pipeline.run(str(src), diarizer, transcriber, aligner, mode=args.mode,
                                max_chunk_s=args.max_chunk, keep_tags=args.keep_tags)
        segments, stats = postprocess(segments, post_cfg)
        logging.getLogger("diarscribe").info(summary(stats))
        out_dir = args.output_dir or src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in output.write_all(segments, out_dir / src.stem, formats):
            print(f"→ {p}", file=sys.stderr)
        trimmed = sum("loop_trimmed" in s.flags for s in segments)
        if trimmed:
            print(f"⚠ {trimmed} segment(s) ont bouclé malgré les relances : texte nettoyé, à relire (marqués ⚠).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
