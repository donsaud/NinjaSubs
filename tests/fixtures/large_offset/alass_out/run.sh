#!/bin/sh
set -e
cd /tmp/lo_mad
for ref in dexter_s08e04_reference valid_constant_offset negative_wrong_episode negative_different_cut negative_drifting; do
  out="out_${ref}.srt"
  rm -f "$out"
  set +e
  alass "fx/${ref}.srt" "fx/dexter_s08e04_target.srt" "$out" --split-penalty 7.0 >/dev/null 2>&1
  rc=$?
  set -e
  if [ -s "$out" ]; then
    n=$(grep -c ' --> ' "$out" || true)
    echo "OK   ref=$ref exit=$rc cues=$n -> $out"
  else
    echo "FAIL ref=$ref exit=$rc (no output)"
  fi
done
