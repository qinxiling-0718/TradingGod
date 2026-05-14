"""Prosperity scoring engine — aggregates factors into dimensional scores then sector scores.

The scoring flow:
    Factors → Dynamic Weights (InfluenceRegistry) → Dimension Scores → Sector Composite Score → Ranking

This is the "景气度评分" module that generates the core signal for sector rotation.

Supports both static weights (score_sectors) and dynamic influence weights (score_sectors_dynamic).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from factors.registry import FactorDimension
from factors.influence import (
    FactorInfluence,
    InfluenceRegistry,
    MetaState,
    profit_cycle_influence,
    macro_cycle_influence,
    capital_flow_influence,
    valuation_influence,
    policy_influence,
    default_influence,
)


class ProsperityLevel(str, Enum):
    """Prosperity level classification."""
    BOOMING = "booming"         # Strong upside momentum
    IMPROVING = "improving"     # Trending up
    STABLE = "stable"           # Sideways
    DECLINING = "declining"     # Trending down
    DISTRESSED = "distressed"   # Strong downside pressure


@dataclass
class DimensionScore:
    """Score for a single prosperity dimension."""
    dimension: FactorDimension
    score: float                  # Aggregated z-score
    momentum: float               # Change over last 4 weeks
    effective_weight: float = 0.0  # Dynamic weight used
    contributing_factors: list[str] = field(default_factory=list)


@dataclass
class SectorProsperityScore:
    """Complete prosperity assessment for one sector."""
    sector_code: str
    sector_name: str
    composite_score: float        # Weighted aggregate
    prosperity_level: ProsperityLevel
    dimension_scores: dict[str, DimensionScore] = field(default_factory=dict)
    rank: int = 0


class ProsperityEngine:
    """Aggregates factor-level expectation gaps into sector prosperity scores.

    Supports two modes:
    1. Static weights: score_sectors() — traditional fixed-weight scoring
    2. Dynamic weights: score_sectors_dynamic() — uses InfluenceRegistry for
       adaptive weighting with fuzzy space constraints.
    """

    # Static dimension weights (fallback for non-dynamic mode)
    DIMENSION_WEIGHTS = {
        FactorDimension.PROFIT_CYCLE: 0.30,
        FactorDimension.MACRO_CYCLE: 0.25,
        FactorDimension.INVENTORY_CYCLE: 0.15,
        FactorDimension.VALUATION_MATCH: 0.15,
        FactorDimension.CAPITAL_FLOW: 0.10,
        FactorDimension.POLICY_FORCE: 0.05,
    }

    # Dimension-level influence functions (used in dynamic mode)
    DIMENSION_INFLUENCES = {
        FactorDimension.PROFIT_CYCLE: profit_cycle_influence(0.30),
        FactorDimension.MACRO_CYCLE: macro_cycle_influence(0.25),
        FactorDimension.INVENTORY_CYCLE: default_influence(0.15),
        FactorDimension.VALUATION_MATCH: valuation_influence(0.15),
        FactorDimension.CAPITAL_FLOW: capital_flow_influence(0.10),
        FactorDimension.POLICY_FORCE: policy_influence(0.05),
    }

    def __init__(
        self,
        influence_registry: Optional[InfluenceRegistry] = None,
    ):
        """Initialize the prosperity engine.

        Args:
            influence_registry: Optional InfluenceRegistry for dynamic weighting.
                                If None, static DIMENSION_WEIGHTS are used.
        """
        self.influence = influence_registry or InfluenceRegistry()
        self._use_dynamic = influence_registry is not None

    # ── Static Mode ────────────────────────────────────────────────

    def score_sectors(
        self,
        factor_scores: pd.DataFrame,
        sector_mapping: dict[str, str],
    ) -> list[SectorProsperityScore]:
        """Compute prosperity scores with STATIC dimension weights.

        Args:
            factor_scores: DataFrame with columns [date, sector, factor_name, gap_zscore].
            sector_mapping: Map from sector_code to sector_name.

        Returns:
            List of SectorProsperityScore, sorted by composite_score descending.
        """
        return self._score_impl(factor_scores, sector_mapping, dynamic=False)

    # ── Dynamic Mode ───────────────────────────────────────────────

    def score_sectors_dynamic(
        self,
        factor_scores: pd.DataFrame,
        sector_mapping: dict[str, str],
        meta_state: Optional[MetaState] = None,
    ) -> list[SectorProsperityScore]:
        """Compute prosperity scores with DYNAMIC influence weights.

        Dimension weights are adjusted based on current meta-factor state
        (volatility regime, cycle position, factor consensus, etc.) and
        clamped within each dimension's fuzzy space [weight_min, weight_max].

        Args:
            factor_scores: DataFrame with columns [date, sector, factor_name, gap_zscore].
            sector_mapping: Map from sector_code to sector_name.
            meta_state: Current meta-factor state. If None, uses neutral state.

        Returns:
            List of SectorProsperityScore, sorted by composite_score descending.
        """
        if meta_state is not None:
            self.influence.set_meta_state(meta_state)

        return self._score_impl(factor_scores, sector_mapping, dynamic=True)

    # ── Implementation ─────────────────────────────────────────────

    def _score_impl(
        self,
        factor_scores: pd.DataFrame,
        sector_mapping: dict[str, str],
        dynamic: bool,
    ) -> list[SectorProsperityScore]:
        """Shared implementation for both static and dynamic scoring."""

        # Compute effective dimension weights
        if dynamic:
            dim_weights = {}
            meta = self.influence.get_meta_state()
            for dim, influence_fn in self.DIMENSION_INFLUENCES.items():
                dim_weights[dim] = influence_fn.effective_weight(meta)
            # Normalize
            total = sum(dim_weights.values())
            if total > 0:
                dim_weights = {k: v / total for k, v in dim_weights.items()}
        else:
            dim_weights = dict(self.DIMENSION_WEIGHTS)

        sectors = {}
        for sector in factor_scores["sector"].unique():
            sec_data = factor_scores[factor_scores["sector"] == sector]

            dim_scores = {}
            composite = 0.0

            for dim, weight in dim_weights.items():
                dim_factors = sec_data[
                    sec_data["factor_name"].apply(
                        lambda n: self._factor_dimension(n) == dim
                    )
                ]
                if dim_factors.empty:
                    continue

                avg_zscore = dim_factors["gap_zscore"].mean()
                dim_scores[dim.value] = DimensionScore(
                    dimension=dim,
                    score=avg_zscore,
                    momentum=0.0,
                    effective_weight=weight,
                    contributing_factors=dim_factors["factor_name"].tolist(),
                )
                composite += weight * avg_zscore

            sectors[sector] = SectorProsperityScore(
                sector_code=sector,
                sector_name=sector_mapping.get(sector, sector),
                composite_score=composite,
                prosperity_level=self._classify_level(composite),
                dimension_scores=dim_scores,
            )

        # Rank sectors
        ranked = sorted(
            sectors.values(),
            key=lambda s: s.composite_score,
            reverse=True,
        )
        for i, s in enumerate(ranked):
            s.rank = i + 1

        return ranked

    def get_dimension_weight_report(self) -> pd.DataFrame:
        """Return a DataFrame comparing base vs effective dimension weights."""
        meta = self.influence.get_meta_state()
        rows = []
        for dim, influence_fn in self.DIMENSION_INFLUENCES.items():
            eff = influence_fn.effective_weight(meta)
            rows.append({
                "dimension": dim.value,
                "base_weight": influence_fn.base_weight,
                "effective": eff,
                "min": influence_fn.weight_min,
                "max": influence_fn.weight_max,
                "drift": eff - influence_fn.base_weight,
            })
        return pd.DataFrame(rows)

    # ── Helpers ────────────────────────────────────────────────────

    def _factor_dimension(self, factor_name: str) -> FactorDimension:
        """Map factor name to dimension. Override for custom mapping."""
        return FactorDimension.PROFIT_CYCLE

    def _classify_level(self, score: float) -> ProsperityLevel:
        """Classify prosperity level from composite z-score."""
        if score > 1.5:
            return ProsperityLevel.BOOMING
        if score > 0.5:
            return ProsperityLevel.IMPROVING
        if score > -0.5:
            return ProsperityLevel.STABLE
        if score > -1.5:
            return ProsperityLevel.DECLINING
        return ProsperityLevel.DISTRESSED
