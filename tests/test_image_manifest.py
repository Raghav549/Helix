import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from helix.image_workflow import load_image_manifest


def test_image_manifest_enforces_review_hash_and_quarantine(tmp_path):
    array = np.zeros((24, 24, 3), dtype=np.uint8)
    array[:, 12:, 0] = 255
    path = tmp_path / 'a.png'; Image.fromarray(array).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    base = {'path': 'a.png', 'sha256': digest, 'caption': 'red shape', 'license': 'MIT',
            'license_evidence': 'LICENSE', 'rights_confirmed': True, 'safety_reviewed': True}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'images': [base, base, {**base, 'license': 'UNKNOWN'},
                                              {**base, 'sha256': '0'*64}, {'path': 'missing.png'}, 'bad record']}))
    images, captions, report = load_image_manifest(str(manifest))
    assert images.shape == (1, 3, 32, 32) and captions == ['red shape']
    reasons = [r['reason'] for r in report['quarantined']]
    assert len(reasons) == 5
    assert any('duplicate' in r for r in reasons)
    assert any('rights' in r for r in reasons)
    assert any('sha256' in r for r in reasons)
    assert any('record must be' in r for r in reasons)
