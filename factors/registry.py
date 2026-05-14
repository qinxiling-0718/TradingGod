"""Factor registry — dynamic factor definition and lifecycle management.

Key design principle: factors are not hardcoded formulas. Each factor is a
pipeline specification: (data_source, transform, rolling_window, aggregation).
This allows the user to define new factors in YAML or Python without touching
the engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd


class FactorDimension(str, Enum):
    """Top-level classification of factor types for prosperity analysis."""

    PROFIT_CYCLE = "profit_cycle"         # 盈利周期
    MACRO_CYCLE = "macro_cycle"           # 宏观周期
    INVENTORY_CYCLE = "inventory_cycle"   # 库存周期
    VALUATION_MATCH = "valuation_match"   # 估值匹配
    CAPITAL_FLOW = "capital_flow"          # 资金行为
    POLICY_FORCE = "policy_force"          # 政策力度


class GapDirection(str, Enum):
    """Expectation gap direction."""

    POSITIVE = "positive"     # 现实 > 预期 (positive surprise)
    NEGATIVE = "negative"     # 现实 < 预期 (negative surprise)
    NEUTRAL = "neutral"       # 接近预期


@dataclass
class FactorDefinition:
    """A single factor's complete definition.

    This is the core data structure that the factor engine operates on.
    Each instance represents one specific factor in the factor library.

    Attributes:
        name: Unique factor identifier (e.g., 'profit_surprise', 'pmi_momentum')
        dimension: High-level category
        description: Human-readable explanation
        data_source: Which adapter/data table provides the raw data
        expected_field: Column name for 'expected' value
        actual_field: Column name for 'actual/realized' value
        lookback_weeks: Rolling window for computation (weeks)
        transform: Optional transform to apply (e.g., 'zscore', 'rank', 'pct_change')
        weight: Default weight when combining into composite score
        min_history_weeks: Minimum data history required for valid computation
    """

    name: str
    dimension: FactorDimension
    description: str
    data_source: str
    expected_field: str
    actual_field: str
    lookback_weeks: int = 12
    optimal_frequency: str = "weekly"   # daily / weekly / monthly
    transform: Optional[str] = None
    weight: float = 1.0
    min_history_weeks: int = 8

    # Raw data cache (populated at compute time)
    raw_data: Optional[pd.DataFrame] = field(default=None, repr=False)


class FactorRegistry:
    """Central registry for all factors.

    Supports:
    - Registering factors from Python code
    - Loading factor definitions from YAML config
    - Listing factors by dimension
    - Validation of factor completeness
    """

    def __init__(self):
        self._factors: dict[str, FactorDefinition] = {}

    def register(self, factor: FactorDefinition) -> None:
        """Register a single factor definition."""
        if factor.name in self._factors:
            raise ValueError(f"Factor '{factor.name}' already registered")
        self._factors[factor.name] = factor

    def unregister(self, name: str) -> None:
        """Remove a factor from the registry."""
        self._factors.pop(name, None)

    def get(self, name: str) -> Optional[FactorDefinition]:
        """Get a factor by name."""
        return self._factors.get(name)

    def list_all(self) -> list[str]:
        """Return all factor names."""
        return list(self._factors.keys())

    def list_by_dimension(self, dimension: FactorDimension) -> list[str]:
        """Return factor names for a given dimension."""
        return [
            name
            for name, f in self._factors.items()
            if f.dimension == dimension
        ]

    def get_weights(self, dimension: Optional[FactorDimension] = None) -> dict[str, float]:
        """Get factor weights, optionally filtered by dimension."""
        factors = (
            {n: f for n, f in self._factors.items() if f.dimension == dimension}
            if dimension
            else self._factors
        )
        total = sum(f.weight for f in factors.values())
        if total == 0:
            return {n: 0.0 for n in factors}

        return {n: f.weight / total for n, f in factors.items()}

    def clear(self) -> None:
        """Remove all registered factors."""
        self._factors.clear()

    def __len__(self) -> int:
        return len(self._factors)

    def __contains__(self, name: str) -> bool:
        return name in self._factors

    def __repr__(self) -> str:
        dims = {}
        for f in self._factors.values():
            dims.setdefault(f.dimension.value, []).append(f.name)
        return f"FactorRegistry({dict(dims)})"
