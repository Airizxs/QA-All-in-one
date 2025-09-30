"""Interactive launcher for common QA tools."""

from __future__ import annotations

from typing import List

from .interactive import run_audit_flow


def _run_seo_checker() -> None:
    run_audit_flow()


def run_menu() -> None:
    while True:
        print("\n==============================")
        print("  Automated QA Command Center  ")
        print("==============================")
        print("  [A] Run SEO Audit")
        print("  [B] Spell Checker (coming soon)")
        print("  [Q] Exit")
        choice = input("Please choose an option [A/B/Q]: ").strip().upper()

        if choice == "A":
            _run_seo_checker()
        elif choice == "B":
            print("Spell Checker is not yet available. Stay tuned!\n")
        elif choice == "Q":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please choose A, B, or Q.\n")


if __name__ == "__main__":  # pragma: no cover - manual launcher
    run_menu()
