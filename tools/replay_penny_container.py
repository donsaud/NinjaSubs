"""Run existing container ALASS on local fixtures, using only temporary files."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REMOTE = '''
import json, sys, time
import app.services.sync.orchestrator  # import first: avoids sync_service circular import
from app.services.sync_service import SubtitleSyncService
data=json.load(sys.stdin)
start=time.perf_counter()
candidate=SubtitleSyncService().sync(data['target'],data['reference'],
    decision_kind='edition', is_series=True)
alass_ms=(time.perf_counter()-start)*1000
print(json.dumps({'candidate':candidate,'alass_ms':alass_ms}))
'''

if __name__ == '__main__':
    root = Path('tests/fixtures/penny')
    data = {'target': (root / 'penny-s01e01-hdtv-ar-2.srt').read_text(encoding='utf-8-sig'),
            'reference': (root / 'penny-s01e01-bluray-aos-ar.srt').read_text(encoding='utf-8-sig')}
    result = subprocess.run(['docker', 'exec', '-i', 'ninjasubs', 'python', '-c', REMOTE],
                            input=json.dumps(data), text=True, encoding='utf-8',
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    report = json.loads(result.stdout)
    candidate = report.pop('candidate')
    if candidate:
        Path('reports/penny-block353-alass.srt').write_text(candidate, encoding='utf-8')
    Path('reports/penny-block353-replay.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
