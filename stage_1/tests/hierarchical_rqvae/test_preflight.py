import numpy as np
from hierarchical_rqvae.preflight import choose_groups, fingerprint


def test_duplicate_groups_cannot_cross_partitions():
    records = [dict(traj_id=i, gps_seq=[[float(i//2), 0]], time_seq=[0]) for i in range(20)]
    chosen, groups = choose_groups(records, 8, 42)
    assert len(groups) == 10
    assert all(i % 2 == 0 for _, i in chosen)
    val, train = chosen[:2], chosen[2:]
    assert not set(k for k,_ in val) & set(k for k,_ in train)
    rest, _ = choose_groups(records, 2, 42, [k for k,_ in chosen])
    assert not set(k for k,_ in rest) & set(k for k,_ in chosen)


def test_fingerprint_ignores_id_but_includes_time():
    a = dict(traj_id=1, gps_seq=[[1, 2]], time_seq=[0])
    assert fingerprint(a) == fingerprint(dict(a, traj_id=5))
    assert fingerprint(a) != fingerprint(dict(a, time_seq=[1]))
