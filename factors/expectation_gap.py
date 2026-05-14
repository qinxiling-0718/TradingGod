"""Expectation Gap Calculator — the core alpha engine.

Philosophy:
    Expectation Gap = Actual (or high-frequency tracked) value - Market Consensus

    Alpha comes from the DIRECTION and MAGNITUDE of changes in the gap:
    - Positive widening gap → positive alpha (reality is beating expectations by more)
    - Negative widening gap → negative alpha
    - Gap narrowing → signal decay

This module provides:
1. Raw gap computation: actual - expected
2. Gap standardization: z-score for cross-factor comparability
3. Gap momentum: change in gap over time (Δ gap)
4. Signal generation: threshold-based buy/sell signals
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from factors.registry import FactorDefinition, GapDirection


@dataclass
class GapResult:
    """Output from a single expectation gap computation."""

    factor_name: str
    dates: np.ndarray
    gap_raw: np.ndarray        # actual - expected (raw difference)
    gap_zscore: np.ndarray     # standardized gap (rolling z-score)
    gap_momentum: np.ndarray   # Δ gap (change over lookback)
    direction: np.ndarray      # GapDirection per observation
    signal: np.ndarray         # -1, 0, +1 signal
    confidence: np.ndarray     # 0.0 - 1.0 signal confidence


class ExpectationGapCalculator:
    """Computes expectation gaps for registered factors.

    Usage:
        registry = FactorRegistry()
        registry.register(some_factor)
        calc = ExpectationGapCalculator(registry)
        result = calc.compute("profit_surprise", actual_df, expected_df)
    """

    def __init__(
        self,
        zscore_window: int = 52,       # 52-week rolling for z-score
        momentum_window: int = 4,       # 4-week gap momentum
        signal_threshold: float = 1.0,  # Z-score > |1.0| triggers signal
        decay_factor: float = 0.85,     # Weight decay for older obs
    ):
        self.zscore_window = zscore_window
        self.momentum_window = momentum_window
        self.signal_threshold = signal_threshold
        self.decay_factor = decay_factor

    def compute(
        self,
        factor: FactorDefinition,
        actual_series: pd.Series,
        expected_series: pd.Series,
    ) -> GapResult:
        """Compute expectation gap for a single factor.

        Args:
            factor: The factor definition.
            actual_series: Time series of actual/realized values.
            expected_series: Time series of expected/consensus values
                             (must align with actual_series by index).

        Returns:
            GapResult with computed gap metrics.
        """
        # Align the two series
        common_idx = actual_series.index.intersection(expected_series.index)
        actual = actual_series.loc[common_idx].astype(float)
        expected = expected_series.loc[common_idx].astype(float)

        if len(actual) < factor.min_history_weeks:
            return self._empty_result(factor.name, len(actual))

        # 1. Raw gap: actual - expected
        gap_raw = (actual - expected).values

        # 2. Standardized gap (rolling z-score)
        gap_zscore = self._rolling_zscore(gap_raw, self.zscore_window)

        # 3. Gap momentum (change in gap)
        gap_momentum = self._compute_momentum(gap_raw, self.momentum_window)

        # 4. Direction classification
        direction = np.array([
            self._classify_direction(z) for z in gap_zscore
        ])

        # 5. Signal generation with confidence
        signal, confidence = self._generate_signal(gap_zscore, gap_momentum)

        return GapResult(
            factor_name=factor.name,
            dates=common_idx.values,
            gap_raw=gap_raw,
            gap_zscore=gap_zscore,
            gap_momentum=gap_momentum,
            direction=direction,
            signal=signal,
            confidence=confidence,
        )

    def _rolling_zscore(self, values: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling z-score with decay weighting."""
        n = len(values)
        z = np.full(n, np.nan)

        for i in range(window - 1, n):
            segment = values[i - window + 1 : i + 1]

            # Apply decay weights
            weights = self.decay_factor ** np.arange(window - 1, -1, -1)
            weighted_mean = np.average(segment, weights=weights)
            weighted_var = np.average((segment - weighted_mean) ** 2, weights=weights)
            weighted_std = np.sqrt(weighted_var)

            if weighted_std > 0:
                z[i] = (values[i] - weighted_mean) / weighted_std

        return z

    def _compute_momentum(self, values: np.ndarray, window: int) -> np.ndarray:
        """Compute the change in gap over a rolling window."""
        n = len(values)
        momentum = np.full(n, np.nan)

        for i in range(window, n):
            momentum[i] = values[i] - values[i - window]

        return momentum

    def _classify_direction(self, z: float) -> str:
        """Classify expectation gap direction from z-score."""
        if np.isnan(z):
            return GapDirection.NEUTRAL.value
        if z > 0.5:
            return GapDirection.POSITIVE.value
        if z < -0.5:
            return GapDirection.NEGATIVE.value
        return GapDirection.NEUTRAL.value

    def _generate_signal(
        self, gap_zscore: np.ndarray, gap_momentum: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate trade signals from gap metrics.

        Signal rules:
        - Gap z-score > +threshold AND momentum > 0 → Strong positive (+1)
        - Gap z-score > +threshold AND momentum <= 0 → Weak positive (+0.5)
        - Gap z-score < -threshold AND momentum < 0 → Strong negative (-1)
        - Gap z-score < -threshold AND momentum >= 0 → Weak negative (-0.5)
        - Within threshold → Neutral (0)

        Confidence is derived from the product of |zscore| and momentum direction.
        """
        n = len(gap_zscore)
        signal = np.zeros(n)
        confidence = np.zeros(n)

        for i in range(n):
            z = gap_zscore[i]
            m = gap_momentum[i]

            if np.isnan(z) or np.isnan(m):
                signal[i] = 0
                confidence[i] = 0
                continue

            abs_z = abs(z)
            sig = 1 if z > 0 else -1

            if abs_z >= self.signal_threshold:
                # Strong signal when momentum confirms
                if (sig > 0 and m > 0) or (sig < 0 and m < 0):
                    signal[i] = sig
                    confidence[i] = min(1.0, abs_z / (self.signal_threshold * 2))
                else:
                    signal[i] = sig * 0.5
                    confidence[i] = min(0.7, abs_z / (self.signal_threshold * 2))

        return signal, confidence

    def _empty_result(self, name: str, length: int) -> GapResult:
        """Return an empty result for insufficient data."""
        empty = np.full(length, np.nan)
        return GapResult(
            factor_name=name,
            dates=np.array([]),
            gap_raw=empty,
            gap_zscore=empty,
            gap_momentum=empty,
            direction=np.array([GapDirection.NEUTRAL.value] * length),
            signal=np.zeros(length),
            confidence=np.zeros(length),
        )


def compute_composite_score(
    results: list[GapResult],
    weights: list[float],
) -> pd.DataFrame:
    """Combine multiple gap results into a composite prosperity score.

    This is the STATIC weight version. For dynamically adjusted weights,
    use compute_composite_dynamic() with an InfluenceRegistry.
    """
    """Combine multiple gap results into a composite prosperity score.

    Args:
        results: List of GapResult from different factors.
        weights: Corresponding weights (will be normalized).

    Returns:
        DataFrame with columns: date, composite_score, composite_confidence,
        weighted_signal.
    """
    if not results:
        return pd.DataFrame()

    # Align all results by date
    all_dates = set()
    for r in results:
        all_dates.update(r.dates)
    all_dates = sorted(all_dates)

    # Normalize weights
    w = np.array(weights) / np.sum(weights)

    # Build weighted composite
    composite_score = np.zeros(len(all_dates))
    composite_confidence = np.zeros(len(all_dates))

    for i, date_val in enumerate(all_dates):
        weighted_sum = 0.0
        conf_sum = 0.0

        for j, result in enumerate(results):
            idx = np.where(result.dates == date_val)[0]
            if len(idx) == 0:
                continue

            k = idx[0]
            if not np.isnan(result.gap_zscore[k]):
                weighted_sum += w[j] * result.gap_zscore[k]
                conf_sum += w[j] * result.confidence[k]

        composite_score[i] = weighted_sum
        composite_confidence[i] = conf_sum

    # Weighted signal (discretized composite)
    weighted_signal = np.where(
        composite_score > 1.0, 1,
        np.where(composite_score < -1.0, -1, 0),
    )

    return pd.DataFrame({
        "date": all_dates,
        "composite_score": composite_score,
        "composite_confidence": composite_confidence,
        "weighted_signal": weighted_signal,
    })


def compute_composite_dynamic(
    results: list[GapResult],
    influence_registry,  # InfluenceRegistry
    meta_state=None,     # MetaState, optional override
) -> pd.DataFrame:
    """Combine gap results using dynamic factor influence weights.

    Unlike compute_composite_score() which uses fixed weights, this uses
    the InfluenceRegistry to compute effective weights based on current
    meta-factor state (volatility regime, cycle position, factor consensus, etc.).

    The weights are clamped within each factor's [weight_min, weight_max] range,
    providing "fuzzy space" protection against overfitting.

    Args:
        results: List of GapResult from different factors.
        influence_registry: InfluenceRegistry with factor influence functions.
        meta_state: Optional MetaState override. If None, uses the registry's
                    current meta state.

    Returns:
        DataFrame with columns: date, composite_score, composite_confidence,
        weighted_signal, and per-factor effective weights.
    """
    if not results:
        return pd.DataFrame()

    if meta_state is None:
        meta_state = influence_registry.get_meta_state()

    # Get dynamic effective weights (clamped to fuzzy space)
    eff_weights = influence_registry.effective_weights_normalized()

    # Map factor names to their effective weights
    w_map = {
        name: eff_weights.get(name, 0.0)
        for name in [r.factor_name for r in results]
    }

    # Align all results by date
    all_dates = set()
    for r in results:
        all_dates.update(r.dates)
    all_dates = sorted(all_dates)

    # Build weighted composite
    composite_score = np.zeros(len(all_dates))
    composite_confidence = np.zeros(len(all_dates))

    for i, date_val in enumerate(all_dates):
        weighted_sum = 0.0
        conf_sum = 0.0
        total_w = 0.0

        for result in results:
            idx = np.where(result.dates == date_val)[0]
            if len(idx) == 0:
                continue

            k = idx[0]
            if not np.isnan(result.gap_zscore[k]):
                w = w_map.get(result.factor_name, 0.0)
                weighted_sum += w * result.gap_zscore[k]
                conf_sum += w * result.confidence[k]
                total_w += w

        composite_score[i] = weighted_sum / total_w if total_w > 0 else 0.0
        composite_confidence[i] = conf_sum / total_w if total_w > 0 else 0.0

    weighted_signal = np.where(
        composite_score > 1.0, 1,
        np.where(composite_score < -1.0, -1, 0),
    )

    return pd.DataFrame({
        "date": all_dates,
        "composite_score": composite_score,
        "composite_confidence": composite_confidence,
        "weighted_signal": weighted_signal,
    })
