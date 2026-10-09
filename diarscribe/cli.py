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
from .asr import Transcriber  # noqa: E402
from .diarize import Diarizer  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
LANGUAGES = ["ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="diarscribe", description="Transcription diarisée (Nemotron-3-Diarization + Cohere Transcribe)")
    ap.add_argument("audio", nargs="+", help="fichier(s) audio/vidéo (tout format lu par ffmpeg)")
    ap.add_argument("-o", "--output-dir", type=Path, help="dossier de sortie (défaut : à côté de l'audio)")
    ap.add_argument("-f", "--formats", default="txt,srt,json", help="formats de sortie parmi txt,srt,json")
    ap.add_argument("-l", "--lang", choices=LANGUAGES, default="fr",
                    help="langue de l'audio (Cohere Transcribe ne la détecte pas) ; écarte aussi les sorties dans une autre écriture")
    ap.add_argument("--mode", choices=["window", "turns"], default="window",
                    help="window : fenêtres continues ≤ 30 s, mots alignés puis attribués au locuteur (plus de contexte) ; "
                         "turns : un appel par tour de parole")
    ap.add_argument("--align-model", help="modèle CTC wav2vec2 pour l'alignement des mots (défaut selon --lang)")
    ap.add_argument("--no-punctuation", action="store_true", help="transcription sans ponctuation ni majuscules")
    ap.add_argument("--keep-tags", action="store_true", help="garder les balises [Silence], [Noise]...")
    ap.add_argument("--batch-size", type=int, default=16, help="tours de parole transcrits par appel au modèle")
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
        ap.error("--max-chunk doit être dans ]0, 35] (au-delà, le modèle redécoupe lui-même l'audio)")

    print("Chargement de Nemotron-3-Diarization...", file=sys.stderr)
    diarizer = Diarizer(device=args.diar_device or args.device)
    print("Chargement de Cohere Transcribe...", file=sys.stderr)
    transcriber = Transcriber(
        device=args.device, dtype=DTYPES[args.dtype], language=args.lang, punctuation=not args.no_punctuation,
        batch_size=args.batch_size, max_wps=args.max_wps,
    )
    aligner = None
    if args.mode == "window":
        print("Chargement de l'aligneur wav2vec2...", file=sys.stderr)
        aligner = Aligner(language=args.lang, device=args.device, model_id=args.align_model)

    for audio in args.audio:
        src = Path(audio)
        segments = pipeline.run(str(src), diarizer, transcriber, aligner, mode=args.mode,
                                max_chunk_s=args.max_chunk, keep_tags=args.keep_tags)
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
