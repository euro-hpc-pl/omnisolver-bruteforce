# SPDX-FileCopyrightText: 2021-2026 The Omnisolver developers
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dtype resolution and problem size validation.

These do not launch any kernel, so they run wherever the extension can be imported.
"""

import numpy as np
import pytest

from omnisolver.bruteforce.gpu.sampler import normalize_dtype, validate_kernel_problem_size


class TestNormalizeDtype:
    @pytest.mark.parametrize("spec", ["float", "float32", "single", "FLOAT", " float "])
    def test_single_precision_names_resolve_to_float32(self, spec):
        """The CLI passes "float" as a plain string, and it has to mean single precision.

        NumPy resolves both "float" and "double" to float64, so passing the raw string through
        would silently disable the single-precision fast path together with its stabilization.
        """
        assert normalize_dtype(spec) is np.float32

    @pytest.mark.parametrize("spec", ["double", "float64"])
    def test_double_precision_names_resolve_to_float64(self, spec):
        assert normalize_dtype(spec) is np.float64

    @pytest.mark.parametrize(
        "spec, expected",
        [
            (np.float32, np.float32),
            (np.float64, np.float64),
            (np.dtype(np.float32), np.float32),
            (np.dtype("float64"), np.float64),
        ],
    )
    def test_numpy_specifications_are_accepted(self, spec, expected):
        assert normalize_dtype(spec) is expected

    @pytest.mark.parametrize("spec", ["float16", "int", np.int32, np.float16, np.complex128])
    def test_unsupported_precisions_are_rejected(self, spec):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            normalize_dtype(spec)


class TestValidateKernelProblemSize:
    @pytest.mark.parametrize("num_variables", [1, 30, 64])
    def test_accepts_sizes_that_fit_the_state_word(self, num_variables):
        validate_kernel_problem_size(num_variables, suffix_size=1)

    @pytest.mark.parametrize("num_variables", [65, 80])
    def test_rejects_sizes_wider_than_the_state_word(self, num_variables):
        with pytest.raises(ValueError, match="64-bit word"):
            validate_kernel_problem_size(num_variables, suffix_size=1)

    @pytest.mark.parametrize("suffix_size", [0, -1, 31])
    def test_rejects_suffix_not_fitting_in_the_problem(self, suffix_size):
        with pytest.raises(ValueError, match="suffix_size"):
            validate_kernel_problem_size(30, suffix_size=suffix_size)

    def test_accepts_suffix_equal_to_the_problem_size(self):
        validate_kernel_problem_size(30, suffix_size=30)
