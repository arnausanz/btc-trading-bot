#!/usr/bin/env python
# scripts/reset_demo.py
"""
Resets demo bot state: deletes all ticks and trades from the DB so every
bot starts fresh with the initial capital on the next run.

Usage:
  python scripts/reset_demo.py              # reset ALL bots (asks for confirmation)
  python scripts/reset_demo.py xgboost      # reset a single bot
  python scripts/reset_demo.py xgboost dca  # reset multiple bots
  python scripts/reset_demo.py --yes        # skip confirmation (for scripts/CI)
"""
import sys

sys.path.append(".")

import argparse

from core.db.demo_repository import DemoRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset demo bot state to initial capital.")
    parser.add_argument(
        "bots",
        nargs="*",
        metavar="BOT_ID",
        help="Bot IDs to reset. If omitted, resets ALL bots.",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    repo = DemoRepository()

    if args.bots:
        target = ", ".join(args.bots)
        scope  = f"bots: {target}"
    else:
        scope  = "ALL bots"

    if not args.yes:
        print(f"\n⚠️  This will permanently delete all ticks and trades for {scope}.")
        print("   The next run will start fresh with the initial capital.\n")
        answer = input("Are you sure? [y/N] ").strip().lower()
        if answer != "y":
            print("Cancelled.")
            sys.exit(0)

    if args.bots:
        for bot_id in args.bots:
            ticks, trades = repo.reset_bot_state(bot_id)
            print(f"  ✓ {bot_id}: deleted {ticks} ticks, {trades} trades.")
    else:
        results = repo.reset_all_states()
        if not results:
            print("  No data found — DB is already clean.")
        else:
            for bot_id, (ticks, trades) in results.items():
                print(f"  ✓ {bot_id}: deleted {ticks} ticks, {trades} trades.")

    print("\nDone. Restart the demo runner to apply.")


if __name__ == "__main__":
    main()
