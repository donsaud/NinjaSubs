# Cached dialogue-alignment regressions

These are the real subtitle bytes already present in the owner's local runtime
cache. No provider request or download was made to acquire them for these tests.
The cache was read, not modified. No video, signed URL, configuration, or
credential is included.

| Episode | Original target cache ID | Target cues | Reference cues |
|---|---|---:|---:|
| S08E02 | `2fc09c4f89a18d3b` (AFG) | 894 | 642 |
| S08E04 | `d010faffc7142fdb` (AFG) | 781 | 562 |
| S08E05 | `5220e5bed50af11f` (ASAP) | 885 | 584 |

`target.srt` and `reference.srt` were copied byte-for-byte. References are the
exact corresponding episode members retained from the d3g season pack whose
provider source digest is `518b87f28a5b9ffe`. Original CRLF line endings remain.

E04/E05 `candidate.srt` files were generated from their target/reference bytes by
`classify_pair` and `build_linear_output`. They preserve original dialogue/order
and propose timestamps only. Their verdicts come from actual correspondence
residuals; the production code contains no episode-specific acceptance rule.
There is no guessed E02 candidate: this reference pair classifies as nonlinear.

The executable checks are in `tests/test_dialogue_alignment_policy.py`.
`docs/dialogue_alignment_refactor.md` records provenance, timing measurements,
scope limitations, lifecycle proof, and the separate available E02 bulk case.

Raw-byte SHA-256 values:

| Episode | File | SHA-256 |
|---|---|---|
| E02 | target | `51b9c3bfcfdc8382db12437454a51b4693e611fd46f475c4702ddd5d26025462` |
| E02 | reference | `f0749e1421e8f1b3da2cdb5afb2ea2fd578cc60b1a9cd2b538e22e92c3b22c42` |
| E04 | target | `30246e82e06d3ee2a0e14ebbffe4d29e0559f52762afff10f8214b117d564ac2` |
| E04 | reference | `5a68255ddba2f9677a785952dc91a853a232560fb6177ce3a549bb09b581f470` |
| E04 | candidate | `8f7259f2cb044910664e498a6d35077118996ba2aed1a0a1f0610748404f8096` |
| E05 | target | `92484b4867d0197496ff8c9f4f92eb6f766d16e4356e070ed89a6576a840bc8b` |
| E05 | reference | `f30a7d8f508e763907e45be26854cf0cd0d1027eb78e475ae94cfbb612af729a` |
| E05 | candidate | `50c526680282f8317feb521212d66329cdd28ce2b801b0f3ff89aef65bce0bbe` |
