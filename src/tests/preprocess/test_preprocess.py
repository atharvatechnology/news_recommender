import pandas as pd
import os

import pytest

from src.preprocess import save_dataframe_to_csv


def test_save_dataframe_to_csv():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    save_dataframe_to_csv(df, "test.csv")
    assert pd.read_csv("data/test.csv").equals(df)


# clean up in fixture
@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    os.remove("data/test.csv")
