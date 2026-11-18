"""Command line interface: sample generation and pipeline runs."""

import argparse
import sys

from .pipeline import Pipeline
from .samplegen import generate_configs, generate_samples


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="workbench",
        description="Video event boundary & trajectory diagnostic workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen-samples", help="generate deterministic samples+configs")
    gen.add_argument("--samples", default="samples")
    gen.add_argument("--configs", default="configs")
    gen.add_argument("--sequences", type=int, default=4)
    gen.add_argument("--frames", type=int, default=40)

    run = sub.add_parser("run", help="run (or resume) the pipeline")
    run.add_argument("--samples", default="samples")
    run.add_argument("--configs", default="configs")
    run.add_argument("--workdir", required=True)
    run.add_argument("--max-concurrency", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "gen-samples":
        names = generate_samples(args.samples, args.sequences, args.frames)
        generate_configs(args.configs)
        print(f"generated {len(names)} sequences under {args.samples}; "
              f"configs under {args.configs}")
        return 0
    pipeline = Pipeline(args.samples, args.configs, args.workdir,
                        max_concurrency=args.max_concurrency)
    return pipeline.run()


if __name__ == "__main__":
    sys.exit(main())
