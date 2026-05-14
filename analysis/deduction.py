"""Logic deduction engine for cross-sector chain reasoning.

Purpose:
    Pure data-driven correlation can miss structural relationships between sectors.
    The deduction engine encodes **industry chain knowledge** so that signals in
    upstream sectors can propagate to downstream expectations.

Two chain graphs are provided:
    1. INDUSTRY_CHAINS — Generic A-share sector chains (28 SW industries)
    2. AI_INDUSTRY_CHAINS — AI-specific supply chain for the core AI investment thesis

The engine uses a directed graph of industry relationships to propagate signals
and cross-validate multi-source evidence.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ChainNode:
    """A node in the industry chain graph."""
    sector: str
    sector_name: str
    upstream: list[str] = field(default_factory=list)
    downstream: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class DeductionResult:
    """Result of a deduction chain."""
    chain: list[str]
    initial_signal: str
    propagated_score: float
    confidence: float


# ── Generic A-share Industry Chains ─────────────────────────────────

INDUSTRY_CHAINS = {
    "real_estate": ChainNode(
        sector="real_estate",
        sector_name="房地产",
        upstream=["finance"],
        downstream=["steel", "building_materials", "home_appliances", "building_decoration"],
    ),
    "steel": ChainNode(
        sector="steel",
        sector_name="钢铁",
        upstream=["mining", "real_estate", "infrastructure"],
        downstream=["auto", "mechanical_equipment", "building_decoration"],
    ),
    "nonferrous": ChainNode(
        sector="nonferrous",
        sector_name="有色金属",
        upstream=["mining"],
        downstream=["electronics", "auto", "electric_equipment", "national_defense"],
    ),
    "auto": ChainNode(
        sector="auto",
        sector_name="汽车",
        upstream=["steel", "nonferrous", "electronics", "chemical"],
        downstream=["leisure_services"],
    ),
    "electronics": ChainNode(
        sector="electronics",
        sector_name="电子",
        upstream=["nonferrous", "chemical"],
        downstream=["computer", "telecom", "media", "auto", "home_appliances"],
    ),
    "electric_equipment": ChainNode(
        sector="electric_equipment",
        sector_name="电气设备",
        upstream=["nonferrous", "steel"],
        downstream=["utilities", "auto"],
    ),
    "mining": ChainNode(
        sector="mining",
        sector_name="采掘",
        upstream=[],
        downstream=["steel", "nonferrous", "utilities", "chemical"],
    ),
}


# ── AI-specific Supply Chain ────────────────────────────────────────
#
# Chain structure for AI industry analysis:
#
#   Layer 1 (Upstream - Materials):
#     nonferrous → rare earth, copper, high-end materials for chips/connectors
#     chemical → electronic-grade chemicals, photoresist, CMP slurry
#
#   Layer 2 (Core - Semiconductors):
#     electronics → chip design, foundry, packaging, HBM memory, storage
#     mechanical_equipment → semiconductor manufacturing equipment
#
#   Layer 3 (Infrastructure - Computing):
#     computer → AI servers, data centers, AI platforms
#     telecom → optical modules (800G/1.6T), high-speed connectivity
#     electric_equipment → power supply, liquid cooling, UPS for data centers
#
#   Layer 4 (Applications):
#     media → AI-generated content, AI marketing, gaming
#     auto → autonomous driving, smart cockpit
#     pharma → AI drug discovery
#     finance → AI fintech, robo-advisor
#

AI_INDUSTRY_CHAINS = {
    # ── Layer 1: Materials ──
    "nonferrous": ChainNode(
        sector="nonferrous",
        sector_name="有色金属",
        upstream=["mining"],
        downstream=["electronics", "electric_equipment", "computer"],
        description="铜/稀土/高端合金 → 芯片互联/散热/连接器",
    ),
    "chemical": ChainNode(
        sector="chemical",
        sector_name="化工",
        upstream=["mining"],
        downstream=["electronics", "mechanical_equipment"],
        description="电子级化学品/光刻胶/CMP抛光液 → 芯片制造",
    ),

    # ── Layer 2: Core Semiconductors ──
    "electronics": ChainNode(
        sector="electronics",
        sector_name="电子(半导体)",
        upstream=["nonferrous", "chemical", "mechanical_equipment"],
        downstream=["computer", "telecom", "auto", "media", "electric_equipment"],
        description="芯片设计/制造/封测/HBM存储 → 所有算力下游",
    ),
    "mechanical_equipment": ChainNode(
        sector="mechanical_equipment",
        sector_name="机械设备(半导体设备)",
        upstream=["nonferrous", "steel"],
        downstream=["electronics"],
        description="刻蚀机/薄膜沉积/检测设备 → 芯片制造",
    ),

    # ── Layer 3: Computing Infrastructure ──
    "computer": ChainNode(
        sector="computer",
        sector_name="计算机(算力/平台)",
        upstream=["electronics", "telecom", "electric_equipment"],
        downstream=["media", "auto", "pharma", "finance", "telecom"],
        description="AI服务器/数据中心/大模型平台 → AI应用",
    ),
    "telecom": ChainNode(
        sector="telecom",
        sector_name="通信(光通信/网络)",
        upstream=["electronics", "computer"],
        downstream=["computer", "media"],
        description="800G/1.6T光模块/高速交换 → 算力互联",
    ),
    "electric_equipment": ChainNode(
        sector="electric_equipment",
        sector_name="电气设备(电力/散热)",
        upstream=["nonferrous", "electronics"],
        downstream=["computer", "telecom"],
        description="液冷/UPS/电力供应 → 数据中心",
    ),

    # ── Layer 4: AI Applications ──
    "media": ChainNode(
        sector="media",
        sector_name="传媒(AI应用)",
        upstream=["computer", "electronics"],
        downstream=[],
        description="AI生成内容/智能营销/游戏AI",
    ),
    "auto": ChainNode(
        sector="auto",
        sector_name="汽车(智能驾驶)",
        upstream=["electronics", "computer", "steel", "nonferrous"],
        downstream=[],
        description="自动驾驶/智能座舱/车规芯片",
    ),
    "pharma": ChainNode(
        sector="pharma",
        sector_name="医药生物(AI制药)",
        upstream=["computer"],
        downstream=[],
        description="AI药物发现/蛋白质预测",
    ),
    "finance": ChainNode(
        sector="finance",
        sector_name="银行(AI金融)",
        upstream=["computer"],
        downstream=["real_estate", "auto"],
        description="AI风控/智能投顾/量化交易",
    ),
    "national_defense": ChainNode(
        sector="national_defense",
        sector_name="国防军工(AI国防)",
        upstream=["electronics", "computer", "telecom"],
        downstream=[],
        description="AI侦察/无人系统/智能弹药",
    ),
}


class DeductionEngine:
    """Propagates signals through industry chains and cross-validates evidence.

    Supports both generic industry chains and AI-specific supply chains.
    """

    def __init__(self, chains: dict[str, ChainNode] = None, chain_type: str = "ai"):
        """Initialize the deduction engine.

        Args:
            chains: Custom chain graph. If None, uses built-in chains based on chain_type.
            chain_type: "ai" for AI supply chain, "generic" for general industry chains.
        """
        if chains is not None:
            self.chains = chains
        elif chain_type == "ai":
            self.chains = AI_INDUSTRY_CHAINS
        else:
            self.chains = INDUSTRY_CHAINS
        self.chain_type = chain_type

    def propagate(
        self,
        source_sector: str,
        source_signal: float,
        depth: int = 3,
    ) -> list[DeductionResult]:
        """Propagate a signal through downstream chains.

        Uses BFS to traverse the directed graph. Signal attenuates with each
        hop: strength *= 0.7^depth, confidence *= 0.8^depth.

        Args:
            source_sector: Origin sector code.
            source_signal: Signal strength (positive = bullish, negative = bearish).
            depth: Max chain depth (hops from source).

        Returns:
            List of DeductionResult for each affected sector.
        """
        results = []
        visited = {source_sector}

        queue = [(source_sector, source_signal, 1.0, 0)]
        chain_path = [source_sector]

        while queue:
            sector, signal, confidence, current_depth = queue.pop(0)

            if current_depth >= depth:
                continue

            node = self.chains.get(sector)
            if node is None:
                continue

            for ds in node.downstream:
                if ds in visited:
                    continue
                visited.add(ds)

                attenuated_signal = signal * (0.7 ** (current_depth + 1))
                attenuated_confidence = confidence * (0.8 ** (current_depth + 1))

                results.append(DeductionResult(
                    chain=chain_path + [ds],
                    initial_signal=source_sector,
                    propagated_score=attenuated_signal,
                    confidence=attenuated_confidence,
                ))

                queue.append((ds, attenuated_signal, attenuated_confidence, current_depth + 1))

        return results

    def propagate_upstream(
        self,
        target_sector: str,
        source_signals: dict[str, float],
        depth: int = 3,
    ) -> list[DeductionResult]:
        """Propagate signals from multiple sources UPSTREAM to a target.

        Instead of asking "what does sector X affect?", this asks
        "what upstream signals are converging on sector X?"

        Args:
            target_sector: The downstream sector to analyze.
            source_signals: Dict of {sector_code: signal_strength} for known signals.
            depth: Max chain depth.

        Returns:
            List of DeductionResult for signals reaching the target.
        """
        results = []

        for source_sector, signal in source_signals.items():
            if source_sector == target_sector:
                continue

            # Try propagating from source. If target is in the propagation
            # results, record the path and strength.
            propagated = self.propagate(source_sector, signal, depth=depth)
            for r in propagated:
                if r.chain[-1] == target_sector:
                    results.append(r)

        return results

    def cross_validate(
        self,
        sector: str,
        direct_signal: float,
        propagated_signals: list[DeductionResult],
    ) -> dict:
        """Cross-validate a direct signal against propagated signals.

        - If direct and propagated signals AGREE in direction → boost confidence 1.5x
        - If they DISAGREE → flag contradiction for investigation
        - If no propagated signals → neutral

        Returns:
            Dict with: sector, direct_signal, propagated_signal, agreement,
            confidence_boost, warning.
        """
        if not propagated_signals:
            return {
                "sector": sector,
                "direct_signal": direct_signal,
                "agreement": "neutral",
                "confidence_boost": 1.0,
                "warning": None,
            }

        propagated_scores = [r.propagated_score for r in propagated_signals]
        avg_propagated = np.mean(propagated_scores)

        if direct_signal * avg_propagated > 0:
            return {
                "sector": sector,
                "direct_signal": direct_signal,
                "propagated_signal": avg_propagated,
                "agreement": "aligned",
                "confidence_boost": 1.5,
                "warning": None,
            }
        elif direct_signal * avg_propagated < 0:
            return {
                "sector": sector,
                "direct_signal": direct_signal,
                "propagated_signal": avg_propagated,
                "agreement": "contradiction",
                "confidence_boost": 0.0,
                "warning": (
                    f"Contradiction: direct={direct_signal:.2f}, "
                    f"chain_avg={avg_propagated:.2f}"
                ),
            }
        else:
            return {
                "sector": sector,
                "direct_signal": direct_signal,
                "propagated_signal": 0.0,
                "agreement": "neutral",
                "confidence_boost": 1.0,
                "warning": None,
            }

    def get_chain_summary(self, sector: str) -> dict:
        """Get the upstream and downstream sectors for a given sector."""
        node = self.chains.get(sector)
        if node is None:
            return {"sector": sector, "found": False}

        return {
            "sector": sector,
            "name": node.sector_name,
            "found": True,
            "upstream": node.upstream,
            "downstream": node.downstream,
            "description": node.description,
        }

    def list_all_sectors(self) -> list[str]:
        """Return all sector codes in the current chain graph."""
        return list(self.chains.keys())
