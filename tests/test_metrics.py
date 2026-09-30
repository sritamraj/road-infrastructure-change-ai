from src.evaluation.metrics import binary_metrics


def test_binary_metrics_perfect_prediction():
    result = binary_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert result["tp"] == 2
    assert result["tn"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] > 0.999
    assert result["recall"] > 0.999
    assert result["iou"] > 0.999


def test_binary_metrics_all_wrong():
    result = binary_metrics([0, 1], [0.9, 0.1])
    assert result["tp"] == 0
    assert result["tn"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["iou"] == 0.0
