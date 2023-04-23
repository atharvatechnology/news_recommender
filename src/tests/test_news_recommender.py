#!/usr/bin/env python

# Example Tests

import pytest
from ..preprocess import get_file_path


def test_get_file_path():
    DATA_PATH = "data"
    assert get_file_path("users.csv") == "data/users.csv"
