"""Analysis Report Generator — prototype Agent output.

Combines PEG factors + metadata annotations into a human-readable
analysis report. This is NOT a dashboard — it's a conversation starter.

Usage:  uv run python scripts/generate_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from factors.peg_factor import compute_peg_factors, aggregate_to_sectors
from factors.metadata_annotator import generate_analysis_report, GrowthNature, PegTrapType


def main():
    print("=" * 65)
    print("TradingGod Analysis Report — AI Supply Chain")
    print("=" * 65)

    # Compute factors
    peg_df = compute_peg_factors(use_revisions=False)
    if peg_df.empty:
        print("No data. Run ingest_tushare.py first.")
        return

    sector_agg = aggregate_to_sectors(peg_df)

    # Build signal dicts for annotation
    signals = []
    for _, r in sector_agg.iterrows():
        signals.append({
            "sector": r["sector"],
            "name": r["sector"],
            "peg": r["avg_peg"],
            "growth": r["avg_growth"],
            "signal": r["signal_score"],
            "n_analysts": int(r["stock_count"]),
            "dispersion": r.get("avg_dispersion", 0),
            "dispersion_trend": "stable",
            "stocks": int(r["stock_count"]),
        })

    annotations = generate_analysis_report(signals)

    # ── Print annotated report ──
    for i, ann in enumerate(annotations):
        sig_char = "+" if ann.signal_direction == "positive" else ("-" if ann.signal_direction == "negative" else "~")

        print(f"\n{'─' * 65}")
        print(f"#{i+1}  {ann.sector_name}  [{sig_char}]  signal={ann.signal_score:+.3f}")
        print(f"{'─' * 65}")

        print(f"  PEG = {ann.peg:.1f}  |  Growth = {ann.consensus_growth:.1%}  |  "
              f"Analysts = {ann.analyst_count}  |  Dispersion = {ann.dispersion:.1%}")

        # Growth nature
        nature_label = {
            GrowthNature.STRUCTURAL: "STRUCTURAL (AI CapEx driven)",
            GrowthNature.CYCLICAL: "CYCLICAL (macro/commodity driven)",
            GrowthNature.HYBRID: "HYBRID (mixed drivers)",
            GrowthNature.UNCERTAIN: "UNCERTAIN (model cannot judge)",
        }[ann.growth_nature]
        print(f"  Growth Nature: {nature_label}")
        if ann.growth_nature_reason:
            print(f"    -> {ann.growth_nature_reason}")

        # PEG Trap Analysis (if active)
        if ann.peg_trap.trap_type.value != "none":
            trap = ann.peg_trap
            print(f"\n  [!] PEG TRAP DETECTED — {trap.headline}")
            print(f"  {'=' * 55}")
            print(f"  SCENARIO A (bear): {trap.scenario_a}")
            print(f"  SCENARIO B (bull): {trap.scenario_b}")
            print(f"  Model lean: {trap.model_lean:.0%} toward Scenario B ({'slight' if abs(trap.model_lean - 0.5) < 0.2 else 'moderate'} bias)")
            print(f"\n  EVIDENCE CHECKLIST (human must answer):")
            for i, check in enumerate(trap.evidence_checklist):
                print(f"    [{i+1}] {check}")

        # Model observations
        print(f"\n  [MODEL OBSERVATIONS]")
        for obs in ann.model_observations:
            print(f"    - {obs}")

        # Narrative alignment
        print(f"\n  [NARRATIVE vs DATA]")
        print(f"    Status: {ann.narrative.value.upper()}")
        print(f"    {ann.narrative_note}")

        # Blind spots
        print(f"\n  [BLIND SPOTS — Model Cannot See]")
        for bs in ann.blind_spots:
            print(f"    >> {bs}")

        # Risk flags
        if ann.risk_flags:
            print(f"\n  [RISK FLAGS]")
            for rf in ann.risk_flags:
                print(f"    ! {rf}")

        # Human questions
        print(f"\n  [QUESTIONS FOR HUMAN JUDGMENT]")
        for q in ann.questions_for_human:
            print(f"    ? {q}")

    # ── Summary matrix ──
    print(f"\n\n{'=' * 65}")
    print("Decision Matrix — Summary")
    print(f"{'=' * 65}")
    print(f"{'Sector':<25s} {'PEG':>6s} {'Growth':>7s} {'Signal':>8s} {'Nature':>12s}")
    print("-" * 65)
    for ann in annotations:
        sig = f"{ann.signal_score:+.3f}"
        print(f"{ann.sector_name:<25s} {ann.peg:>5.1f}  {ann.consensus_growth:>6.1%}  {sig:>8s}  {ann.growth_nature.value:>12s}")

    print()
    print("=" * 65)
    print("Note: This is a data-driven analysis with model annotations.")
    print("Final decisions require human judgment on structural vs cyclical,")
    print("external risk assessment, and growth sustainability evaluation.")
    print("=" * 65)


if __name__ == "__main__":
    main()
