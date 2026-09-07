import json
import pytest
from hierarchical_rqvae.run import verify_preflight
from hierarchical_rqvae.preflight import sha256, atomic_json


def test_preflight_corruption_and_leakage_fail_closed(tmp_path):
    cohort=tmp_path/'cohort'; cohort.mkdir()
    data=cohort/'dummy'; data.write_text('frozen')
    manifest={'cohort':str(cohort),'cohort_hashes':{'dummy':sha256(data)},'train_ids':[1],'validation_ids':[2]}
    atomic_json(tmp_path/'manifest.json',manifest)
    atomic_json(tmp_path/'complete.json',{'manifest_sha256':sha256(tmp_path/'manifest.json')})
    assert verify_preflight(tmp_path)['train_ids']==[1]
    data.write_text('tampered')
    with pytest.raises(ValueError,match='artifact hash'):
        verify_preflight(tmp_path)
    data.write_text('frozen')
    manifest['validation_ids']=[1]
    atomic_json(tmp_path/'manifest.json',manifest)
    atomic_json(tmp_path/'complete.json',{'manifest_sha256':sha256(tmp_path/'manifest.json')})
    with pytest.raises(ValueError,match='leakage'):
        verify_preflight(tmp_path)
