import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'paper'))

from evaluate_paper_models import aurc, brier, ece, mae, mse, pcc


def assert_close(actual, expected, tol=1e-6):
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f'{actual} != {expected}')


def main():
    assert_close(pcc([1, 2, 3], [1, 2, 3]), 1.0)
    assert_close(pcc([1, 2, 3], [3, 2, 1]), -1.0)
    assert_close(mse([1, 2, 3], [1, 1, 5]), (0 + 1 + 4) / 3)
    assert_close(mae([1, 2, 3], [1, 1, 5]), (0 + 1 + 2) / 3)
    perfect_conf = [0.05, 0.95]
    perfect_target = [0.0, 1.0]
    if ece(perfect_conf, perfect_target, bins=2) >= 0.1:
        raise AssertionError('ECE should be small for well separated confidence bins.')
    if brier(perfect_conf, perfect_target) >= 0.01:
        raise AssertionError('Brier score should be small for near-correct confidence.')
    if not math.isfinite(aurc([0.9, 0.1, 0.8], [0.1, 0.8, 0.2])):
        raise AssertionError('AURC must be finite.')
    print('paper metric tests passed')


if __name__ == '__main__':
    main()
