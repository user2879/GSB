from pathlib import Path

from arg_review.samplegen import SEED, generate_sample_tree


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "samples"
    paths = generate_sample_tree(root)
    print(f"deterministic seed: {SEED}")
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")
