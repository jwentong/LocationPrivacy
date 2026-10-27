"""Compare generated paper tables against frozen numerical references."""
from pathlib import Path
import numpy as np
import pandas as pd
root = Path(__file__).resolve().parent
reference = root / "reference_results/basic_v1"
actual = root / "Experiments/results/basic_v1"
# parameters.csv contains runtime and script hashes, intentionally not compared.
files = [p for p in sorted(reference.glob("*.csv")) if p.name != "parameters.csv"]
for path in files:
    expected = pd.read_csv(path)
    observed = pd.read_csv(actual / path.name)
    assert list(expected.columns) == list(observed.columns), path.name
    assert expected.shape == observed.shape, path.name
    for col in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[col]):
            np.testing.assert_allclose(observed[col].to_numpy(dtype=float), expected[col].to_numpy(dtype=float), rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=f"{path.name}:{col}")
        else:
            pd.testing.assert_series_equal(observed[col], expected[col], check_names=False)
    print(f"PASS {path.name}: {len(expected)} rows")
print(f"Verified {len(files)} numerical result tables against basic_v1 references.")
