from independent_latent_sid.diagnose_code_replacement import first_wrong, score


def test_first_wrong_is_event_aligned_and_layer_specific():
    target = [[1, 2, 3, 4], [5, 6, 7, 8]]
    generated = [[1, 9, 3, 4], [5, 6, 0, 8]]
    assert first_wrong(target, generated, 0) is None
    assert first_wrong(target, generated, 1) == 0
    assert first_wrong(target, generated, 2) == 1
    assert first_wrong(target, generated, None) == 0


def test_four_component_score_is_zero_for_exact_match():
    stats = {
        "travel_distance_km": 2.0,
        "mean_segment_distance_km": 0.1,
        "radius_km": 0.4,
        "mean_point_to_road_km": 0.02,
    }
    assert score(stats, stats) == 0
