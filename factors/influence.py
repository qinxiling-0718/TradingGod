"""Dynamic Factor Influence Engine — fuzzy space constraints over fixed weights.

Philosophy:
    Factor weights are NOT constants. They shift with market regime, but within
    bounded intervals ("fuzzy space"). This gives adaptability without overfitting.

    effective_weight = clamp(base + Σ(sensitivity × meta_factor), min, max)

Meta-factors are "factors on factors" — they describe the environment in which
a factor operates, not the market itself. Examples: volatility regime, cycle
position, factor consensus, data freshness.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


# ── Meta-Factor Definitions ─────────────────────────────────────────

class MetaFactor(str, Enum):
    """Environmental conditions that modulate factor influence."""

    VOLATILITY_REGIME = "volatility_regime"     # High vol → flow factor less reliable
    CYCLE_POSITION = "cycle_position"           # Where in macro cycle: early/mid/late
    FACTOR_CONSENSUS = "factor_consensus"       # Multi-factor agreement level
    POLICY_INTENSITY = "policy_intensity"       # Policy window active
    DATA_FRESHNESS = "data_freshness"           # How recent is the underlying data
    TREND_STRENGTH = "trend_strength"           # Trend clarity (vs choppy/sideways)


class CyclePosition(str, Enum):
    """Macro cycle phase."""
    EARLY = "early"         # Recovery beginning
    MID = "mid"             # Expansion
    LATE = "late"           # Peak / overheating
    CONTRACTION = "contraction"  # Downturn


@dataclass
class MetaState:
    """Snapshot of current meta-factor values.

    All values are normalized to roughly [-1, 1] range for consistent scaling.
    """
    volatility_regime: float = 0.0       # -1=extremely low, 0=normal, 1=extremely high
    cycle_position: str = "mid"          # early / mid / late / contraction
    factor_consensus: float = 0.0        # -1=all disagree, 0=mixed, 1=all aligned
    policy_intensity: float = 0.0        # 0=no policy window, 1=active policy push
    data_freshness: float = 0.0          # 0=stale, 1=just updated
    trend_strength: float = 0.0          # -1=strong downtrend, 0=choppy, 1=strong uptrend

    def to_dict(self) -> dict:
        return {
            "volatility_regime": self.volatility_regime,
            "cycle_position": self.cycle_position,
            "factor_consensus": self.factor_consensus,
            "policy_intensity": self.policy_intensity,
            "data_freshness": self.data_freshness,
            "trend_strength": self.trend_strength,
        }

    @classmethod
    def neutral(cls) -> "MetaState":
        """Return a neutral meta-state (all zeros, mid cycle)."""
        return cls()


# ── Factor Influence ────────────────────────────────────────────────

@dataclass
class FactorInfluence:
    """Dynamic influence function for a single factor.

    The influence is:
        effective = base_weight + Σ(sensitivity_i × meta_factor_i)
        clamped to [weight_min, weight_max]

    The clamping is the "fuzzy space" protection — weights can adapt but
    cannot escape their boundary. This prevents overfitting to any single
    market regime.

    Attributes:
        base_weight: Center-point weight (used when all meta factors are neutral).
        weight_min: Hard floor — effective weight never goes below this.
        weight_max: Hard ceiling — effective weight never goes above this.
        meta_sensitivities: Mapping from MetaFactor → sensitivity coefficient.
            Positive = factor gets more weight when meta factor is high.
            Negative = factor gets less weight when meta factor is high.
    """

    base_weight: float
    weight_min: float
    weight_max: float
    meta_sensitivities: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.weight_min > self.weight_max:
            raise ValueError(
                f"weight_min ({self.weight_min}) must be <= weight_max ({self.weight_max})"
            )
        if not (self.weight_min <= self.base_weight <= self.weight_max):
            raise ValueError(
                f"base_weight ({self.base_weight}) must be in [{self.weight_min}, {self.weight_max}]"
            )

    def effective_weight(self, meta_state: MetaState) -> float:
        """Compute the effective weight given current meta-factor state.

        Args:
            meta_state: Current environmental conditions.

        Returns:
            Effective weight, clamped to [weight_min, weight_max].
        """
        w = self.base_weight
        meta_dict = meta_state.to_dict()

        for key, sensitivity in self.meta_sensitivities.items():
            state_val = meta_dict.get(key, 0.0)

            # For categorical meta factors, convert to numeric
            if isinstance(state_val, str):
                state_val = self._encode_categorical(key, state_val)

            w += sensitivity * state_val

        return max(self.weight_min, min(self.weight_max, w))

    def _encode_categorical(self, key: str, value: str) -> float:
        """Encode categorical meta factors to numeric for weighting."""
        if key == "cycle_position":
            return {
                "early": 0.5,
                "mid": 0.0,
                "late": -0.3,
                "contraction": -1.0,
            }.get(value, 0.0)
        return 0.0

    @classmethod
    def fixed(cls, weight: float) -> "FactorInfluence":
        """Create a fixed influence (no adaptation, no fuzzy space).

        This is a convenience for factors that should not adapt.
        """
        return cls(
            base_weight=weight,
            weight_min=weight,
            weight_max=weight,
        )


# ── Influence Registry ──────────────────────────────────────────────

class InfluenceRegistry:
    """Maps factor names to their FactorInfluence definitions.

    Separate from FactorRegistry — factors define WHAT to compute,
    InfluenceRegistry defines HOW MUCH to trust each factor right now.
    """

    def __init__(self):
        self._influences: dict[str, FactorInfluence] = {}
        self._meta_state: MetaState = MetaState.neutral()

    def register(self, factor_name: str, influence: FactorInfluence) -> None:
        """Register a factor's influence function."""
        self._influences[factor_name] = influence

    def get(self, factor_name: str) -> Optional[FactorInfluence]:
        """Get influence function for a factor."""
        return self._influences.get(factor_name)

    def set_meta_state(self, state: MetaState) -> None:
        """Update current meta-factor state."""
        self._meta_state = state

    def get_meta_state(self) -> MetaState:
        """Get current meta-factor state."""
        return self._meta_state

    def effective_weights(self) -> dict[str, float]:
        """Compute current effective weights for all registered factors.

        Returns:
            Mapping from factor_name → effective_weight (clamped).
        """
        return {
            name: inf.effective_weight(self._meta_state)
            for name, inf in self._influences.items()
        }

    def effective_weights_normalized(self) -> dict[str, float]:
        """Compute effective weights normalized to sum to 1.0."""
        raw = self.effective_weights()
        total = sum(raw.values())
        if total == 0:
            return {k: 0.0 for k in raw}
        return {k: v / total for k, v in raw.items()}

    def list_weights_with_bounds(self) -> pd.DataFrame:
        """Return a DataFrame showing base, current effective, min, max for each factor."""
        rows = []
        for name, inf in self._influences.items():
            eff = inf.effective_weight(self._meta_state)
            rows.append({
                "factor": name,
                "base_weight": inf.base_weight,
                "effective": eff,
                "min": inf.weight_min,
                "max": inf.weight_max,
                "drift": eff - inf.base_weight,
            })
        return pd.DataFrame(rows)

    def __contains__(self, name: str) -> bool:
        return name in self._influences

    def __len__(self) -> int:
        return len(self._influences)

    def __repr__(self) -> str:
        return f"InfluenceRegistry({len(self._influences)} factors, meta={self._meta_state})"


# ── Pre-built Influence Templates ───────────────────────────────────

def default_influence(weight: float, dimension: str = "") -> FactorInfluence:
    """Build a sensible default influence with 50% fuzzy bandwidth.

    The weight can drift ±25% from base. This gives adaptability without
    letting any single regime dominate.
    """
    half_band = weight * 0.25
    return FactorInfluence(
        base_weight=weight,
        weight_min=max(0.02, weight - half_band),
        weight_max=min(0.50, weight + half_band),
        meta_sensitivities={
            MetaFactor.FACTOR_CONSENSUS.value: weight * 0.15,
            MetaFactor.DATA_FRESHNESS.value: weight * 0.10,
        },
    )


def profit_cycle_influence(weight: float = 0.30) -> FactorInfluence:
    """Profit cycle factors: most reliable in mid-cycle, stronger when data is fresh."""
    return FactorInfluence(
        base_weight=weight,
        weight_min=0.18,
        weight_max=0.42,
        meta_sensitivities={
            MetaFactor.CYCLE_POSITION.value: 0.05,    # Slight boost mid-cycle
            MetaFactor.DATA_FRESHNESS.value: 0.08,    # Fresh earnings = more trust
            MetaFactor.FACTOR_CONSENSUS.value: 0.04,   # Boost when others agree
        },
    )


def macro_cycle_influence(weight: float = 0.25) -> FactorInfluence:
    """Macro cycle factors: stronger at cycle extremes (early/contraction),
    weaker mid-cycle when micro dominates."""
    return FactorInfluence(
        base_weight=weight,
        weight_min=0.12,
        weight_max=0.40,
        meta_sensitivities={
            MetaFactor.CYCLE_POSITION.value: -0.08,   # Stronger at extremes
            MetaFactor.VOLATILITY_REGIME.value: 0.05, # Volatile macro = pay attention
            MetaFactor.FACTOR_CONSENSUS.value: 0.05,
        },
    )


def capital_flow_influence(weight: float = 0.10) -> FactorInfluence:
    """Flow factors: decay rapidly in high volatility (noise dominates)."""
    return FactorInfluence(
        base_weight=weight,
        weight_min=0.02,   # Can nearly zero out in extreme vol
        weight_max=0.18,
        meta_sensitivities={
            MetaFactor.VOLATILITY_REGIME.value: -0.06,  # HIGH vol → LESS weight
            MetaFactor.TREND_STRENGTH.value: 0.04,       # Clear trend → more weight
            MetaFactor.DATA_FRESHNESS.value: 0.03,
        },
    )


def valuation_influence(weight: float = 0.15) -> FactorInfluence:
    """Valuation: slow-moving, degrade with stale data, amplify at extremes."""
    return FactorInfluence(
        base_weight=weight,
        weight_min=0.08,
        weight_max=0.25,
        meta_sensitivities={
            MetaFactor.DATA_FRESHNESS.value: 0.05,
            MetaFactor.CYCLE_POSITION.value: 0.04,   # More relevant at extremes
        },
    )


def policy_influence(weight: float = 0.05) -> FactorInfluence:
    """Policy: usually dormant, spikes during policy windows."""
    return FactorInfluence(
        base_weight=weight,
        weight_min=0.0,          # Can go to zero when no policy activity
        weight_max=0.20,          # Large spike during active policy windows
        meta_sensitivities={
            MetaFactor.POLICY_INTENSITY.value: 0.12,  # HUGE sensitivity
        },
    )
