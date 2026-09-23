import numpy as np

from hierarchical_rqvae.diagnostics import progress_gps_errors
from independent_latent_sid.diagnose_code_replacement_gps import progress_ade, trip_features


def test_ade_matches_existing_diagnostic_alignment():
    real = np.array([[-8.66, 41.16], [-8.65, 41.17], [-8.64, 41.16]])
    generated = np.array([[-8.66, 41.16], [-8.655, 41.166], [-8.65, 41.17], [-8.64, 41.16]])
    expected = progress_gps_errors(real, generated)["matched_progress_ade_m"]
    assert abs(progress_ade(real, generated) - expected) < 1e-10


def test_trip_features_respect_actual_points_and_route():
    real = np.array([[-8.66, 41.16], [-8.65, 41.17], [-8.64, 41.16]])
    result = trip_features({"route": [1, 2, 3, 4]}, real)
    assert result["points"] == 3
    assert result["events"] == 1
    assert result["route_roads"] == 4
    assert result["endpoint_km"] > 0
