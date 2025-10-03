"""Interactive launcher for common QA tools."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .interactive import run_audit_flow


def _run_seo_checker() -> None:
    run_audit_flow()


def _run_spell_checker() -> None:
    """Launch the standalone spell checker script from the project root."""
    root_dir = Path(__file__).resolve().parents[2]
    script_path = root_dir / "Spell_checker" / "spell_checker.py"

    if not script_path.exists():
        print("Spell Checker script not found. Please ensure Spell_checker/spell_checker.py exists.\n")
        return

    try:
        subprocess.run([sys.executable, str(script_path)], check=True, cwd=root_dir)
    except subprocess.CalledProcessError as exc:
        print(f"Spell Checker exited with an error (code {exc.returncode}).\n")


def run_menu() -> None:
    while True:
        print("\n==============================")
        print("  Automated QA Command Center  ")
        print("==============================")
        print("  [A] Run SEO Audit")
        print("  [B] Spell Checker")
        print("  [Q] Exit")
        choice = input("Please choose an option [A/B/Q]: ").strip().upper()

        if choice == "A":
            _run_seo_checker()
        elif choice == "B":
            _run_spell_checker()
        elif choice == "Q":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please choose A, B, or Q.\n")


if __name__ == "__main__":  # pragma: no cover - manual launcher
    run_menu()
