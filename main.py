"""
main.py — Entry point.  Run this file to execute the full pipeline:

    python main.py
    python main.py --credentials /path/to/sa.json
    python main.py --drug Semaglutide
    python main.py --no-upload
    python main.py --no-clinical
    python main.py --no-marketed-approval

All CLI flags documented in cli.py.
"""

from cli import build_parser
from pipeline import combine

if __name__ == "__main__":
    parser = build_parser()
    args, unknown = parser.parse_known_args()

    # Support "python main.py --Semaglutide" (drug name as its own flag)
    stray_drug_flags = [u for u in unknown if u.startswith("--") and not u.startswith("---")]
    if stray_drug_flags:
        if len(stray_drug_flags) > 1:
            parser.error(
                f"Multiple unrecognized flags given, expected at most one drug name: {stray_drug_flags}"
            )
        leftover = [u for u in unknown if u not in stray_drug_flags]
        if leftover:
            parser.error(f"Unrecognized arguments: {leftover}")
        if args.drug:
            parser.error(
                f"Drug given both via --drug {args.drug!r} and {stray_drug_flags[0]!r}; use only one."
            )
        args.drug = stray_drug_flags[0].lstrip("-")
    elif unknown:
        parser.error(f"Unrecognized arguments: {unknown}")

    if not args.output:
        args.output = (
            f"combined_{args.drug.replace(' ', '_').lower()}.xlsx"
            if args.drug
            else "combined_master_loe.xlsx"
        )

    combine(args)
