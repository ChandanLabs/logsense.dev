#!/usr/bin/env python3
"""
demo.py — BharatRank CLI Demo
=================================
Runs the full pipeline and renders a rich terminal output.
Usage:  python demo.py [--jd JD001] [--top-k 10] [--save]
"""
import argparse
import sys
import os

# ── ANSI colours ────────────────────────────────────────────────────────────
R  = "\033[0m"   # reset
B  = "\033[1m"   # bold
G  = "\033[92m"  # green
Y  = "\033[93m"  # yellow
C  = "\033[96m"  # cyan
W  = "\033[97m"  # white
MG = "\033[95m"  # magenta
BL = "\033[94m"  # blue
DM = "\033[2m"   # dim

def bar(value: float, width: int = 22) -> str:
    filled = int(round(value * width))
    return G + "█" * filled + DM + "░" * (width - filled) + R

def score_colour(v: float) -> str:
    if v >= 0.80: return G
    if v >= 0.60: return Y
    return "\033[91m"

def print_banner():
    print(f"\n{B}{C}{'═'*66}{R}")
    print(f"{B}{C}{'  🇮🇳  BharatRank — Candidate Intelligence Engine':^66}{R}")
    print(f"{B}{DM}{'  Multi-Signal Agentic Ranking System v1.0':^66}{R}")
    print(f"{B}{C}{'═'*66}{R}\n")

def print_jd_header(jd_title: str, total: int):
    print(f"{B}{W}📋  Job:{R} {jd_title}")
    print(f"{B}{W}👥  Pool:{R} {total} candidates evaluated\n")
    print(f"{C}{'─'*66}{R}")
    print(f"{B}{W}  Running BharatRank Pipeline...{R}")
    print(f"{C}{'─'*66}{R}")

def print_result(entry: dict, idx: int):
    rank      = entry["rank"]
    name      = entry.get("candidate_name", "?")
    loc       = entry.get("location", "")
    inst      = entry.get("institution", "")
    role      = entry.get("current_role", "")
    yrs       = entry.get("years_experience", 0)
    score     = entry.get("adjusted_score", entry.get("final_score", 0))
    signals   = entry.get("signals", {})
    skills    = entry.get("top_skills", [])[:5]
    is_gem    = entry.get("is_hidden_gem", False)
    gem_txt   = f"  {MG}{B}💎 HIDDEN GEM{R}" if is_gem else ""
    report    = entry.get("match_report", "")
    sc        = score_colour(score)

    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")

    print(f"\n  {B}┌{'─'*60}┐{R}")
    print(f"  {B}│{R}  {B}{medal}  {W}{name:<30}{R}{gem_txt}")
    print(f"  {B}│{R}  {DM}{role}{R}")
    print(f"  {B}│{R}  {DM}📍 {loc}  •  🎓 {inst}  •  {yrs}y exp{R}")
    print(f"  {B}│{R}")
    print(f"  {B}│{R}  {B}BharatScore:{R}  {sc}{B}{score:.4f}{R}  {bar(score)}")
    print(f"  {B}│{R}")

    labels = {
        "semantic_fit":    "Semantic Fit   ",
        "career_velocity": "Career Velocity",
        "skill_recency":   "Skill Recency  ",
        "signal_strength": "Signal Strength",
        "role_alignment":  "Role Alignment ",
    }
    weights = entry.get("weights", {})
    for key, label in labels.items():
        v  = signals.get(key, 0)
        w  = weights.get(key, 0)
        vc = score_colour(v)
        print(f"  {B}│{R}    {DM}{label}{R}  {vc}{v:.3f}{R}  {bar(v, 14)}  {DM}×{w}{R}")

    print(f"  {B}│{R}")
    print(f"  {B}│{R}  {B}Skills:{R} {', '.join(skills)}")

    # Word-wrap match report
    if report:
        words = report.split()
        line, lines = [], []
        for w_ in words:
            if sum(len(x)+1 for x in line) + len(w_) > 55:
                lines.append(" ".join(line)); line = [w_]
            else:
                line.append(w_)
        if line: lines.append(" ".join(line))
        print(f"  {B}│{R}")
        print(f"  {B}│{R}  {B}{BL}💡 Why:{R}")
        for ln in lines:
            print(f"  {B}│{R}     {ln}")

    print(f"  {B}└{'─'*60}┘{R}")

def main():
    parser = argparse.ArgumentParser(description="BharatRank CLI Demo")
    parser.add_argument("--jd",     default=None,  help="JD ID to rank (e.g. JD001). Default: all.")
    parser.add_argument("--top-k",  type=int, default=10, help="Top-K candidates (default 10)")
    parser.add_argument("--save",   action="store_true",  help="Save output JSON files")
    args = parser.parse_args()

    print_banner()

    from core.ranker import BharatRanker
    ranker = BharatRanker(verbose=True)
    ranker.load_data()

    jd_ids = [args.jd] if args.jd else [jd["id"] for jd in ranker.jds]

    all_results = []
    for jd_id in jd_ids:
        jd_obj = ranker._jds_map.get(jd_id)
        if not jd_obj:
            print(f"{Y}⚠️  JD '{jd_id}' not found. Available: {list(ranker._jds_map.keys())}{R}")
            continue

        print_jd_header(f"{jd_obj['role_title']} @ {jd_obj['company']}", len(ranker.candidates))
        result = ranker.rank(jd_id=jd_id, top_k=args.top_k)
        all_results.append(result)

        total_gems = sum(1 for r in result["results"] if r.get("is_hidden_gem"))
        print(f"\n{B}{G}{'─'*66}{R}")
        print(f"{B}{G}  🏆  TOP {len(result['results'])} CANDIDATES — {jd_obj['role_title']}{R}")
        print(f"{B}{G}  💎  Hidden Gems detected: {total_gems}{R}")
        print(f"{B}{G}{'─'*66}{R}")

        for entry in result["results"]:
            print_result(entry, entry["rank"])

        if args.save:
            path = ranker.save_output(result)
            print(f"\n  {G}💾 Saved → {path}{R}")

    # Summary stats
    print(f"\n{B}{C}{'═'*66}{R}")
    print(f"{B}{C}  📊  SUMMARY{R}")
    print(f"{B}{C}{'═'*66}{R}")
    total_gems = sum(
        sum(1 for r in res["results"] if r.get("is_hidden_gem"))
        for res in all_results
    )
    print(f"  JDs processed  : {len(all_results)}")
    print(f"  Total candidates: {len(ranker.candidates)}")
    print(f"  Hidden Gems surfaced: {total_gems}")
    print(f"  Scoring formula: 0.35×Semantic + 0.20×Velocity + 0.20×Recency + 0.15×Signals + 0.10×Alignment")

    if args.save:
        sub_path = ranker.save_submission(all_results)
        print(f"\n  {G}{B}🎯 Submission file → {sub_path}{R}")

    print(f"\n{B}{C}{'═'*66}{R}\n")

if __name__ == "__main__":
    main()
