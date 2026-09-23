import pytest

from independent_latent_sid.diagnose_sid_errors import classify


def test_missing_and_extra_events_are_kept_separate():
    truth = [[1, 2, 3, 4], [5, 6, 7, 8]]
    short = classify(truth, [[1, 2, 0, 4]])
    assert short["layer_wrong"] == [1, 1, 2, 1]
    assert short["tuple_correct"] == 0
    assert short["missing_events"] == 1
    assert short["extra_events"] == 0
    assert short["first_error"] == [0, 3]

    long = classify(truth, truth + [[9, 9, 9, 9]])
    assert long["layer_wrong"] == [0, 0, 0, 0]
    assert long["tuple_correct"] == 2
    assert long["extra_events"] == 1


def test_malformed_code_shape_fails():
    with pytest.raises(ValueError):
        classify([[1, 2, 3, 4]], [[1, 2, 3]])
