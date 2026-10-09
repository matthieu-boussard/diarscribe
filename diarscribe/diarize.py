"""Speaker diarization with nvidia/Nemotron-3-Diarization (Sortformer, up to 8 speakers)."""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

from .chunking import Turn

log = logging.getLogger(__name__)

MODEL_ID = "nvidia/Nemotron-3-Diarization"
SAMPLE_RATE = 16_000


class Diarizer:
    def __init__(self, device: str = "mps", model_id: str = MODEL_ID):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(model_id, dtype=torch.float32)
        self.device = device
        try:
            self.model.to(device)
        except (RuntimeError, AssertionError) as e:
            log.warning("Diarisation : %s indisponible (%s), bascule sur CPU", device, e)
            self.device = "cpu"
            self.model.to("cpu")
        self.model.eval()
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    @torch.inference_mode()
    def __call__(self, audio: np.ndarray, streaming_threshold_s: float = 600.0, min_speaker_s: float = 2.0) -> list[Turn]:
        """``audio`` must be mono at ``self.sampling_rate``. Long files use chunked streaming."""
        duration = len(audio) / self.sampling_rate
        try:
            segments = self._streaming(audio) if duration > streaming_threshold_s else self._offline(audio)
        except (RuntimeError, NotImplementedError) as e:
            if self.device == "cpu":
                raise
            log.warning("Diarisation sur %s a échoué (%s), nouvel essai sur CPU", self.device, e)
            self.device = "cpu"
            self.model.to("cpu")
            return self(audio, streaming_threshold_s, min_speaker_s)
        turns = [Turn(float(s["Start"]), float(s["End"]), int(s["Speaker"])) for s in segments]
        # Drop phantom speakers (a few stray frames) so they cannot capture short utterances.
        talk: dict[int, float] = {}
        for t in turns:
            talk[t.speaker] = talk.get(t.speaker, 0.0) + t.duration
        if phantoms := {spk for spk, d in talk.items() if d < min_speaker_s}:
            log.info("Locuteurs fantômes ignorés : %s", sorted(phantoms))
        return [t for t in turns if t.speaker not in phantoms]

    def _offline(self, audio: np.ndarray) -> list[dict]:
        inputs = self.processor(audio, sampling_rate=self.sampling_rate).to(self.device, dtype=self.model.dtype)
        logits = self.model(**inputs).logits  # (1, frames, 8), one frame / 10 ms
        return self.processor.extract_speaker_dict(logits, inputs.attention_mask)[0]

    def _streaming(self, audio: np.ndarray) -> list[dict]:
        p = self.processor
        p.set_streaming_mode("low_latency")
        # Chunks overlap: each carries its own frames plus look-ahead frames, but the frame cursor only
        # advances by `num_mel_frames_per_step`; `audio_chunk_start` maps that cursor back to a sample.
        bounds = [(0, min(len(audio), p.num_samples_first_audio_chunk))]
        mel_idx = p.num_mel_frames_per_step
        while (start := p.audio_chunk_start(mel_idx)) + p.num_samples_per_audio_chunk <= len(audio):
            bounds.append((start, start + p.num_samples_per_audio_chunk))
            mel_idx += p.num_mel_frames_per_step
        if start < len(audio) and bounds[-1][1] < len(audio):
            bounds.append((start, len(audio)))

        logits, speaker_cache = [], None
        for i, (a, b) in enumerate(bounds):
            flags = {"is_first_audio_chunk": i == 0, "is_last_audio_chunk": i == len(bounds) - 1}
            inputs = p(audio[a:b], sampling_rate=self.sampling_rate, is_streaming=True, **flags)
            out = self.model(**inputs.to(self.device, dtype=self.model.dtype), speaker_cache=speaker_cache)
            speaker_cache = out.speaker_cache
            logits.append(out.logits)
        return p.extract_speaker_dict(torch.cat(logits, dim=1))[0]
