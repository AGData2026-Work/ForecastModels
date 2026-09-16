import pandas as pd
import pytest

from check_regression import append_run_history


def _paired(mae_by_h: dict) -> pd.DataFrame:
    return pd.DataFrame({"h": list(mae_by_h.keys()), "challenger_MAE": list(mae_by_h.values())})


class TestAppendRunHistory:
    def test_creates_file_on_first_call(self, tmp_path):
        history = tmp_path / "run_history.csv"
        append_run_history(history, "buildA", "GRU", "unconditional", _paired({4: 10.0, 13: 20.0}))
        assert history.exists()
        df = pd.read_csv(history)
        assert len(df) == 2
        assert set(df["h"]) == {4, 13}

    def test_appends_rather_than_overwrites(self, tmp_path):
        history = tmp_path / "run_history.csv"
        append_run_history(history, "buildA", "GRU", "unconditional", _paired({4: 10.0}))
        append_run_history(history, "buildA", "GRU", "unconditional", _paired({4: 10.5}))
        df = pd.read_csv(history)
        assert len(df) == 2
        assert list(df["challenger_MAE"]) == [10.0, 10.5]


def _run_check_regression(history_path, threshold_pct, capsys):
    import sys
    from check_regression import main
    old_argv = sys.argv
    sys.argv = ["check_regression.py", "--history", str(history_path),
               "--threshold-pct", str(threshold_pct)]
    try:
        main()
    finally:
        sys.argv = old_argv
    return capsys.readouterr().out


class TestCheckRegressionFlagging:
    def test_flags_a_large_swing(self, tmp_path, capsys):
        history = tmp_path / "run_history.csv"
        append_run_history(history, "buildA", "GRU", "unconditional", _paired({4: 10.0}))
        append_run_history(history, "buildA", "GRU", "unconditional", _paired({4: 15.0}))  # +50%
        out = _run_check_regression(history, 10.0, capsys)
        assert "buildA GRU unconditional h=4" in out
        assert "+50.0%" in out

    def test_does_not_flag_a_small_move(self, tmp_path, capsys):
        history = tmp_path / "run_history.csv"
        append_run_history(history, "buildA", "GRU", "unconditional", _paired({4: 10.0}))
        append_run_history(history, "buildA", "GRU", "unconditional", _paired({4: 10.2}))  # +2%
        out = _run_check_regression(history, 10.0, capsys)
        assert "buildA" not in out
        assert "no (build, model, convention, h) moved" in out

    def test_first_run_alone_is_not_flagged(self, tmp_path, capsys):
        history = tmp_path / "run_history.csv"
        append_run_history(history, "buildA", "GRU", "unconditional", _paired({4: 999.0}))
        out = _run_check_regression(history, 10.0, capsys)
        assert "buildA" not in out

    def test_missing_history_file_does_not_error(self, tmp_path, capsys):
        out = _run_check_regression(tmp_path / "does_not_exist.csv", 10.0, capsys)
        assert "no history file" in out
