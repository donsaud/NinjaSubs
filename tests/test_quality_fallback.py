"""Quality-aware Large-Offset fallback policy.

The two user-confirmed Mad Men failures used a cached WEB-DL BTN reference
against the HMAX WEB-DL WADU video. Both were weak-binding and low-confidence
(p95 3957/2546 ms in production; 4045/2684 ms on the cached bytes) and both
were delivered by the Large-Offset override, incorrectly.

These tests pin the narrow policy:

* weak/unknown/conflict binding guarded corrections are provisional, not
  immediately deliverable;
* strong/exact binding guarded corrections keep the confirmed Into the Wild /
  Dexter behavior;
* normally verified results are never deferred, even with weak binding;
* non-Large-Offset decisions are untouched.
"""

from __future__ import annotations

import pytest

from app.services.sync.quality_fallback import assess_quality_fallback


def _assess(**overrides):
    base = dict(
        serving_state="alass_corrected_large_offset",
        binding_state="weak",
        would_serve_without_override=False,
        sync_state="unverified",
        verification="unknown",
        rejection_reason="low_confidence",
        p95_ms=3957.0,
        mad_ms=484.0,
        structural_similarity=0.76,
    )
    base.update(overrides)
    return assess_quality_fallback(**base)


def test_mad_men_dbo_weak_low_confidence_is_deferred():
    result = _assess()
    assert result.defer is True
    assert result.reason == "weak_binding_guarded_result_requires_better_evidence"
    assert "binding=weak" in result.summary
    assert "p95=3957.0" in result.summary


def test_mad_men_sa89_weak_low_confidence_is_deferred():
    result = _assess(p95_ms=2546.0, mad_ms=416.5, structural_similarity=0.871)
    assert result.defer is True


@pytest.mark.parametrize("binding", ["unknown", "conflict"])
def test_unknown_or_conflicting_binding_is_also_deferred(binding):
    result = _assess(binding_state=binding)
    assert result.defer is True


def test_into_the_wild_strong_binding_is_still_served():
    result = _assess(binding_state="strong", p95_ms=2511.0, mad_ms=0.0,
                     structural_similarity=0.798)
    assert result.defer is False
    assert result.reason == "strong_binding_guarded_result"


def test_dexter_strong_binding_is_still_served():
    # Measured on the confirmed Dexter S08E01 HDTV-ASAP vs d3g case:
    # strong release-group binding, constant movement, p95 above 2000.
    result = _assess(binding_state="strong", p95_ms=4735.0, mad_ms=0.0,
                     structural_similarity=0.745, rejection_reason="low_confidence")
    assert result.defer is False
    assert result.reason == "strong_binding_guarded_result"


def test_normally_verified_weak_binding_is_not_deferred():
    result = _assess(binding_state="weak", would_serve_without_override=True,
                     sync_state="verified_resynced", verification="verified",
                     rejection_reason=None, p95_ms=0.0)
    assert result.defer is False
    assert result.reason == "normal_verification_accepted"


@pytest.mark.parametrize(
    "state",
    ["original", "synchronized", "alass_corrected_constant_shift"],
)
def test_non_large_offset_serving_states_are_untouched(state):
    result = _assess(serving_state=state)
    assert result.defer is False
    assert result.reason == "not_guarded_large_offset"


def test_fallback_budget_defaults_are_bounded():
    from app.config import settings

    assert settings.QUALITY_FALLBACK_ENABLED is True
    assert 1 <= settings.QUALITY_FALLBACK_MAX_ATTEMPTS <= 3
    assert 0 < settings.QUALITY_FALLBACK_MAX_SECONDS <= 15.0
