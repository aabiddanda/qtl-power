"""Testing module for GWAS power calculations."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qtl_power.lmm import Lmm_GxE


@given(
    a=st.floats(
        min_value=0, max_value=1, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    d=st.integers(min_value=1, max_value=1000),
    ncp=st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False),
)
def test_llr_power(a, d, ncp):
    """Test calculation of log-likelihood ratio calculation."""
    obj = Lmm_GxE()
    obj.llr_power(alpha=a, df=d, ncp=ncp)


# @given(
#     n=st.integers(min_value=1),
#     p=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
#     beta=st.floats(
#         min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False
#     ),
#     r2=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
# )
# def test_ncp_quant(n, p, beta, r2):
#     """Test that the non-centrality parameter is calculatable."""
#     obj = GwasQuant()
#     obj.ncp_quant(n=n, p=p, beta=beta, r2=r2)


# @given(
#     n=st.integers(min_value=1),
#     p=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
#     beta=st.floats(
#         min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False
#     ),
#     r2=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
#     alpha=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
# )
# def test_quant_trait_power(n, p, beta, r2, alpha):
#     """Test the function to obtain power under a quantitative model."""
#     obj = GwasQuant()
#     power = obj.quant_trait_power(n, p, beta, r2, alpha)
#     assert (power >= 0) & (power <= 1)
