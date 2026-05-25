"""Metadata Annotator — adds context layer to factor outputs.

This module does NOT change factor scores. It annotates them with:
  - What the model SEES (data-driven observations)
  - What the model CAN'T SEE (structural/cyclical distinction, external shocks)
  - What needs HUMAN JUDGMENT (growth quality, narrative validity)

This is the bridge between "quant software" and "analyst agent".
"""

from dataclasses import dataclass, field
from enum import Enum


class GrowthNature(str, Enum):
    """The hardest problem — is growth structural or cyclical?"""
    STRUCTURAL = "structural"       # Secular trend (AI, energy transition)
    CYCLICAL = "cyclical"           # Mean-reverting (commodities, shipping)
    HYBRID = "hybrid"               # Structural + cyclical overlap
    UNCERTAIN = "uncertain"         # Model cannot judge


class PegTrapType(str, Enum):
    """Classification of PEG anomaly situations."""
    NONE = "none"                          # No trap detected
    LOW_PEG_VALUE_TRAP = "value_trap"       # Low PEG could be cyclical peak
    LOW_PEG_STRUCTURAL_WINDOW = "structural_window"  # Low PEG = E growing faster than P
    HIGH_PEG_OVERPRICED = "overpriced"      # High PEG = market front-running expectations
    HIGH_PEG_HIDDEN_QUALITY = "hidden_quality"  # High PEG = market correctly pricing moat


class NarrativeAlignment(str, Enum):
    CONFIRMED = "confirmed"         # Narrative matches data direction
    DIVERGENT = "divergent"         # Data contradicts narrative
    LEADING = "leading"             # Data is ahead of narrative
    LAGGING = "lagging"             # Narrative is ahead of data


@dataclass
class PegTrapAnalysis:
    """Structured analysis of a PEG anomaly — is it a trap or an opportunity?"""
    trap_type: PegTrapType = PegTrapType.NONE
    headline: str = ""               # One-line diagnosis
    scenario_a: str = ""             # Bear case (value trap / overpriced)
    scenario_b: str = ""             # Bull case (structural window / hidden quality)
    # Evidence checklist — human needs to answer these
    evidence_checklist: list[str] = field(default_factory=list)
    # How much the model leans toward A vs B (0.0 = pure A, 1.0 = pure B)
    model_lean: float = 0.5


@dataclass
class SectorAnnotation:
    """Metadata annotation for a single sector at a point in time."""

    sector: str
    sector_name: str

    # Core signal
    peg: float
    consensus_growth: float
    signal_score: float
    signal_direction: str  # "positive" / "negative" / "neutral"
    # Hybrid PEG
    revenue_growth: float = 0.0
    profit_quality: float = 0.5
    peg_signal: float = 0.0
    prg_signal: float = 0.0

    # What the model sees
    model_observations: list[str] = field(default_factory=list)

    # Growth quality assessment
    growth_nature: GrowthNature = GrowthNature.UNCERTAIN
    growth_nature_reason: str = ""

    # PEG trap detection (NEW)
    peg_trap: PegTrapAnalysis = field(default_factory=PegTrapAnalysis)

    # What the model CAN'T see (blind spots)
    blind_spots: list[str] = field(default_factory=list)

    # Narrative vs data
    narrative: NarrativeAlignment = NarrativeAlignment.CONFIRMED
    narrative_note: str = ""

    # Consensus quality
    analyst_count: int = 0
    dispersion: float = 0.0
    dispersion_trend: str = ""  # "narrowing" / "widening" / "stable"

    # Risk flags
    risk_flags: list[str] = field(default_factory=list)

    # Human judgment needed
    questions_for_human: list[str] = field(default_factory=list)


def annotate_sector(
    sector: str,
    name: str,
    peg: float,
    growth: float,
    signal: float,
    n_analysts: int = 0,
    dispersion: float = 0.0,
    dispersion_trend: str = "stable",
    stocks: int = 0,
    revenue_growth: float = 0.0,
    profit_quality: float = 0.5,
    peg_signal: float = 0.0,
    prg_signal: float = 0.0,
) -> SectorAnnotation:
    """Generate a complete sector annotation from factor data.

    This function codifies the "analysis method" — the heuristics and
    judgment rules that turn raw numbers into annotated insights.
    """

    direction = "positive" if signal > 0.1 else ("negative" if signal < -0.1 else "neutral")

    ann = SectorAnnotation(
        sector=sector,
        sector_name=name,
        peg=peg,
        consensus_growth=growth,
        signal_score=signal,
        signal_direction=direction,
        revenue_growth=revenue_growth,
        profit_quality=profit_quality,
        peg_signal=peg_signal,
        prg_signal=prg_signal,
        analyst_count=n_analysts,
        dispersion=dispersion,
        dispersion_trend=dispersion_trend,
    )

    # ── Model observations ──
    ann.model_observations = _build_observations(peg, growth, signal, n_analysts)

    # ── Growth nature assessment ──
    ann.growth_nature, ann.growth_nature_reason = _assess_growth_nature(sector, growth)

    # ── PEG trap detection (NEW) ──
    ann.peg_trap = _detect_peg_trap(sector, peg, growth, signal, dispersion)

    # ── Blind spots ──
    ann.blind_spots = _identify_blind_spots(sector)
    # If PEG trap exists, add trap-specific blind spots
    if ann.peg_trap.trap_type != PegTrapType.NONE:
        ann.blind_spots.append(
            f"PEG TRAP ACTIVE: {ann.peg_trap.headline}. "
            f"Model cannot distinguish {ann.peg_trap.scenario_a[:40]}... vs {ann.peg_trap.scenario_b[:40]}..."
        )

    # ── Narrative alignment ──
    ann.narrative, ann.narrative_note = _assess_narrative(sector, peg, growth, signal)

    # ── Risk flags ──
    ann.risk_flags = _identify_risks(sector, peg, growth, dispersion)

    # ── Questions for human ──
    ann.questions_for_human = _generate_questions(
        sector, peg, growth, signal, ann.growth_nature, ann.peg_trap
    )

    return ann


# ── Annotation rule library ────────────────────────────────────────

def _build_observations(peg: float, growth: float, signal: float, n: int) -> list[str]:
    obs = []
    if peg < 1.0:
        obs.append(f"PEG={peg:.1f}, Growth still undervalued relative to price")
    elif peg < 2.0:
        obs.append(f"PEG={peg:.1f}, Reasonable valuation for growth level")
    else:
        obs.append(f"PEG={peg:.1f}, Market is pricing significant growth premium")

    if growth > 0.35:
        obs.append(f"Growth={growth:.1%}, Exceptional — top decile")
    elif growth > 0.20:
        obs.append(f"Growth={growth:.1%}, Strong — above average")
    else:
        obs.append(f"Growth={growth:.1%}, Moderate — needs acceleration narrative")

    if signal > 0.3:
        obs.append("Signal: BUY — PEG + Growth combination attractive")
    elif signal > 0:
        obs.append("Signal: WEAK BUY — marginal attractiveness")
    elif signal > -0.3:
        obs.append("Signal: HOLD/NEUTRAL — no clear edge")
    else:
        obs.append("Signal: AVOID — expensive relative to growth")

    if n > 0:
        obs.append(f"Analyst coverage: {n} analysts")

    return obs


# ── PEG Trap Detection ───────────────────────────────────────────────

def _detect_peg_trap(
    sector: str, peg: float, growth: float, signal: float, dispersion: float
) -> PegTrapAnalysis:
    """Detect PEG anomalies that require human judgment to classify.

    Low PEG (< 0.8) can mean EITHER:
      (A) Value trap: cyclical peak, earnings about to mean-revert down
      (B) Structural window: E is growing faster than P, market mispricing

    High PEG (> 3.0) with high growth can mean EITHER:
      (A) Overpriced: market front-running expectations, little upside
      (B) Hidden quality: market correctly pricing deep moat / scarcity premium
    """

    # ── Low PEG Trap ──
    if peg < 0.8 and growth > 0.15:
        trap_type = PegTrapType.LOW_PEG_VALUE_TRAP
        headline = f"PEG={peg:.2f} is extremely low — trap or opportunity?"
        scenario_a = (
            f"VALUE TRAP (cyclical peak): {growth:.1%} growth is one-time. "
            f"When growth mean-reverts to 10-15%, PEG jumps to {peg * growth / 0.12:.1f}-{peg * growth / 0.15:.1f}. "
            f"Current low PE reflects market correctly anticipating a cycle downturn."
        )
        scenario_b = (
            f"STRUCTURAL WINDOW (growth underpriced): {growth:.1%} growth is structural. "
            f"Earnings are growing FASTER than price — PE is being pulled down by E, not pushed down by P. "
            f"Market is pricing this like a cyclical but the business has structurally changed."
        )

        # Evidence checklist — human answers tip the balance
        evidence = [
            "Capacity: full utilization and expanding, or idle capacity exists?",
            "Order book: locked 3+ months forward, or spot-dependent?",
            "ASP trend: unit price rising (mix shift to higher value), flat, or falling?",
            "End demand: structural (AI CapEx, energy transition) or cyclical (inventory restock, commodity upswing)?",
            "Supply side: competitors also expanding? Is there a barrier to entry?",
            "Historical pattern: has this sector sustained >30% growth for 3+ years before, or always mean-reverted?",
        ]

        # Model leans slightly toward value trap in cyclical sectors, structural in tech
        cyclical_sectors = {"sw_nonferrous", "sw_chemical", "sw_auto"}
        if sector in cyclical_sectors:
            model_lean = 0.35  # lean toward value trap
        else:
            model_lean = 0.55  # slight lean toward structural (AI chain)

        return PegTrapAnalysis(
            trap_type=trap_type,
            headline=headline,
            scenario_a=scenario_a,
            scenario_b=scenario_b,
            evidence_checklist=evidence,
            model_lean=model_lean,
        )

    # ── High PEG Trap ──
    if peg > 3.0 and growth > 0.20:
        trap_type = PegTrapType.HIGH_PEG_OVERPRICED
        headline = f"PEG={peg:.2f} is high — overpriced or quality premium?"
        scenario_a = (
            f"OVERPRICED: {growth:.1%} growth is already fully priced at PEG={peg:.2f}. "
            f"Market has front-run the growth story. Even if growth delivers, "
            f"upside is limited because expectations are so high. "
            f"Any growth deceleration = sharp PEG expansion = painful drawdown."
        )
        scenario_b = (
            f"HIDDEN QUALITY: {growth:.1%} growth understates the true earnings power. "
            f"Consensus estimates are stale or conservative. "
            f"The sector has structural advantages (monopoly, IP, regulation) "
            f"that justify a permanent premium — high PEG is rational, not frothy."
        )

        evidence = [
            "Earnings surprise history: does this sector consistently beat estimates?",
            "Barrier to entry: is there a durable moat (patents, licenses, scale, network effects)?",
            "Growth trajectory: is consensus growth accelerating or decelerating?",
            "Institutional positioning: are long-only funds overweight or underweight?",
            "Short interest: are shorts betting against the growth story?",
        ]

        return PegTrapAnalysis(
            trap_type=trap_type,
            headline=headline,
            scenario_a=scenario_a,
            scenario_b=scenario_b,
            evidence_checklist=evidence,
            model_lean=0.5,  # model truly cannot judge quality premium
        )

    return PegTrapAnalysis(trap_type=PegTrapType.NONE)


# ── Annotation rule library ────────────────────────────────────────

def _assess_growth_nature(sector: str, growth: float) -> tuple[GrowthNature, str]:
    """Heuristic assessment of whether growth is structural or cyclical.

    THIS IS A RULE-BASED APPROXIMATION. Human override is expected.
    """
    structural_sectors = [
        "sw_semiconductor", "sw_electronics", "sw_computer_equipment",
        "sw_telecom_equipment", "sw_software", "sw_it_services",
    ]
    cyclical_sectors = [
        "sw_nonferrous", "sw_chemical", "sw_auto",
    ]
    hybrid_sectors = [
        "sw_media", "sw_pharma", "sw_national_defense",
        "sw_mechanical_equipment", "sw_electric_equipment",
    ]

    if sector in structural_sectors:
        return GrowthNature.STRUCTURAL, "AI supply chain — growth tied to secular CapEx cycle, not inventory cycle"
    elif sector in cyclical_sectors:
        return GrowthNature.CYCLICAL, "Commodity/manufacturing — sensitive to macro cycle and capacity utilization"
    elif sector in hybrid_sectors:
        if growth > 0.25:
            return GrowthNature.HYBRID, "Mixed structural+cyclical drivers — AI component growing, legacy component cyclical"
        else:
            return GrowthNature.CYCLICAL, " Growth below structural threshold — likely cyclical-dominant"
    return GrowthNature.UNCERTAIN, ""


def _identify_blind_spots(sector: str) -> list[str]:
    """Identify what the model CANNOT see for this sector."""
    blind = ["Model cannot distinguish structural vs cyclical growth — human judgment needed"]

    blinds_by_sector = {
        "sw_semiconductor": [
            "Geopolitical risk: US-China chip sanctions trajectory",
            "Domestic substitution timeline vs technical capability gap",
            "Fab capacity expansion pace (requires industry insider knowledge)",
        ],
        "sw_telecom_equipment": [
            "Customer concentration: NA cloud vendors > 60% of revenue",
            "Supply chain risk: single-source optical chips from US/Japan",
            "Tariff/trade policy: direct exposure to trade war escalation",
        ],
        "sw_electronics": [
            "Consumer electronics cycle overlay on AI structural growth",
            "Inventory cycle at downstream OEMs (double-ordering risk)",
        ],
        "sw_computer_equipment": [
            "AI software monetization unclear — hardware demand may precede software revenue",
            "GPU allocation politics (US export controls on H100/B200)",
        ],
        "sw_media": [
            "AI content regulation risk — policy can reset the entire sector",
            "Monetization model unproven for most AI-native products",
        ],
        "sw_nonferrous": [
            "Commodity price: supply-side shocks dominate demand narrative",
            "Global macro sensitivity — Fed policy overrides sector fundamentals",
        ],
    }

    blind.extend(blinds_by_sector.get(sector, []))
    return blind


def _assess_narrative(sector: str, peg: float, growth: float, signal: float) -> tuple[NarrativeAlignment, str]:
    """Compare market narrative with data."""
    if signal > 0.3 and growth > 0.25:
        if peg < 1.5:
            return NarrativeAlignment.LEADING, "Data is ahead of narrative — market still underpricing the growth trajectory"
        else:
            return NarrativeAlignment.CONFIRMED, "Both narrative and data point positive — but PEG suggests narrative is priced in"
    elif signal < -0.3:
        if growth > 0.20:
            return NarrativeAlignment.DIVERGENT, f"Good growth ({growth:.1%}) but expensive (PEG={peg:.1f}) — market may be over-pricing certainty"
        else:
            return NarrativeAlignment.CONFIRMED, f"Weak growth ({growth:.1%}) — narrative and data agree on caution"
    else:
        return NarrativeAlignment.LAGGING, f"Data is neutral — narrative may be ahead of reality"


def _identify_risks(sector: str, peg: float, growth: float, dispersion: float) -> list[str]:
    risks = []

    if peg > 3.0:
        risks.append("High PEG: valuation already prices in significant future growth — little room for disappointment")
    if dispersion > 0.5:
        risks.append(f"High analyst dispersion ({dispersion:.1%}): consensus may not be reliable")
    if growth < 0.15:
        risks.append("Low growth: sector may lack AI exposure or is in structural decline")

    sector_risks = {
        "sw_semiconductor": ["Policy risk: semiconductor is a strategic sector — government intervention can distort pricing"],
        "sw_auto": ["EV transition pace uncertainty", "Price war risk in domestic EV market"],
        "sw_chemical": ["Environmental regulation risk", "Capacity expansion cycle may overshoot"],
    }
    risks.extend(sector_risks.get(sector, []))
    return risks


def _generate_questions(
    sector: str, peg: float, growth: float, signal: float,
    nature: GrowthNature, peg_trap: PegTrapAnalysis,
) -> list[str]:
    """Generate questions that need human judgment."""
    questions = []

    # ── Growth nature questions ──
    if nature == GrowthNature.STRUCTURAL:
        questions.append("Is the structural driver (AI CapEx) accelerating, stable, or decelerating?")
    elif nature == GrowthNature.CYCLICAL:
        questions.append(f"Where are we in the cycle? Is {growth:.1%} growth sustainable or mean-reverting?")
    elif nature == GrowthNature.HYBRID:
        questions.append(f"What proportion of {growth:.1%} growth is structural vs cyclical? Is the structural share growing?")

    # ── PEG trap questions (highest priority — these demand immediate human attention) ──
    if peg_trap.trap_type == PegTrapType.LOW_PEG_VALUE_TRAP:
        questions.append(f"[PEG TRAP: {peg_trap.headline}]")
        # Add the top 3 most discriminating evidence questions
        priority_checks = peg_trap.evidence_checklist[:4]
        for check in priority_checks:
            questions.append(f"  CHECK: {check}")
        questions.append(
            f"  VERDICT: Does this look more like (A) {peg_trap.scenario_a[:60]}... "
            f"or (B) {peg_trap.scenario_b[:60]}...?"
        )

    elif peg_trap.trap_type == PegTrapType.HIGH_PEG_OVERPRICED:
        questions.append(f"[PEG TRAP: {peg_trap.headline}]")
        priority_checks = peg_trap.evidence_checklist[:4]
        for check in priority_checks:
            questions.append(f"  CHECK: {check}")
        questions.append("  VERDICT: Is the premium justified by durable competitive advantage, or is it froth?")

    # ── Standard PEG questions (when no trap) ──
    if peg_trap.trap_type == PegTrapType.NONE:
        if peg > 2.5 and signal < 0:
            questions.append(f"PEG={peg:.1f} is high — could the market be RIGHT to price a premium?")
        if peg < 1.0 and signal > 0:
            questions.append(f"PEG={peg:.1f} is very low — market missing something, or hidden risk?")

    questions.append("What external event (trade war, Fed, regulation) could invalidate this signal within 4 weeks?")

    return questions


# ── Batch annotation generator ──────────────────────────────────────

def generate_analysis_report(
    sector_signals: list[dict],
) -> list[SectorAnnotation]:
    """Generate full annotated report for all sectors.

    Args:
        sector_signals: List of dicts with keys:
            sector, name, peg, growth, signal, n_analysts, dispersion, dispersion_trend

    Returns:
        List of SectorAnnotation, sorted by signal_score descending.
    """
    annotations = []
    for s in sector_signals:
        ann = annotate_sector(
            sector=s.get("sector", "unknown"),
            name=s.get("name", s.get("sector", "")),
            peg=s.get("peg", 0),
            growth=s.get("growth", 0),
            signal=s.get("signal", 0),
            n_analysts=s.get("n_analysts", 0),
            dispersion=s.get("dispersion", 0),
            dispersion_trend=s.get("dispersion_trend", "stable"),
            stocks=s.get("stocks", 0),
            revenue_growth=s.get("revenue_growth", 0),
            profit_quality=s.get("profit_quality", 0.5),
            peg_signal=s.get("peg_signal", 0),
            prg_signal=s.get("prg_signal", 0),
        )
        annotations.append(ann)

    return sorted(annotations, key=lambda a: a.signal_score, reverse=True)
