"""Offline real-fixture verification benchmark (no downloads or runtime caches)."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from app.services.sync import unified_verification as verifier
from app.services.sync.candidate_reconstruction import reconstruct_with_original_dialogue
from tests.test_unified_verification import (
    DEX_TARGET,
    ROVERS_CANDIDATE,
    ROVERS_REFERENCE,
    WHIP_REFERENCE,
    WHIP_TARGET,
    _true_candidate,
)


def run() -> dict:
    root = Path('tests/fixtures/penny')
    read = lambda path: path.read_text(encoding='utf-8-sig')  # noqa: E731
    cases = {
        'penny': (read(root / 'penny-s01e01-hdtv-ar.srt'),
                  read(root / 'penny-s01e01-bluray-aos-ar.srt'),
                  read(root / 'penny-s01e01-alass-aos-output.srt')),
        'penny_block353_original': (read(root / 'penny-s01e01-hdtv-ar-2.srt'),
                                    read(root / 'penny-s01e01-bluray-aos-ar.srt'),
                                    read(root / 'penny-s01e01-hdtv-ar-2.srt')),
        'penny_block353_transformed': (
            # Production path: pristine original + raw ALASS bytes, so the
            # benchmark exercises offset transfer (not the sanitized triple).
            read(root / 'penny-s01e01-hdtv-ar-2.srt'),
            read(root / 'penny-s01e01-bluray-aos-ar.srt'),
            read(root / 'penny-s01e01-block353-alass-output.srt')),
        'dexter': (read(DEX_TARGET), read(ROVERS_REFERENCE), read(ROVERS_CANDIDATE)),
        'whiplash': (read(WHIP_TARGET), read(WHIP_REFERENCE),
                     _true_candidate(read(WHIP_TARGET), 1.04277053, -8707.9)),
    }
    output = {}
    for name, (target, reference, candidate) in cases.items():
        start = time.perf_counter()
        rebuilt = reconstruct_with_original_dialogue(target, candidate) or candidate
        reconstruction_ms = (time.perf_counter() - start) * 1000
        times = []
        for _ in range(4):
            start = time.perf_counter()
            verdict = verifier.verify_candidate(target, reference, rebuilt)
            times.append((time.perf_counter() - start) * 1000)
        output[name] = {
            'reconstruction_ms': round(reconstruction_ms, 2),
            'reconstructed': rebuilt != candidate,
            'first_verification_ms': round(times[0], 2),
            'warm_verification_median_ms': round(statistics.median(times[1:]), 2),
            'verdict': verdict.proposed_verdict,
            'groups': verdict.independent_evidence_count,
            'residuals': verdict.residual_statistics,
            'reasons': verdict.failure_reasons,
        }
    output['verifier_lines'] = len(Path(verifier.__file__).read_text().splitlines())
    output['scope'] = 'Offline verification only; excludes ALASS, HTTP, and reference discovery.'
    return output


if __name__ == '__main__':
    result = json.dumps(run(), indent=2)
    Path(sys.argv[1]).write_text(result + '\n', encoding='utf-8')
    print(result)
