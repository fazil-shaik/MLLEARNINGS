import argparse
import json
import sys

from orchestrator import run_roundtable


def main():
    parser = argparse.ArgumentParser(description="Run a multi-agent Roundtable debate.")
    parser.add_argument("topic", type=str, help="The decision or problem to debate.")
    parser.add_argument(
        "--rounds", type=int, default=1, help="Number of debate rounds (default: 1)."
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Optional path to save the full result as JSON."
    )
    args = parser.parse_args()

    result = run_roundtable(topic=args.topic, rounds=args.rounds, verbose=True)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved full transcript to {args.save}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)