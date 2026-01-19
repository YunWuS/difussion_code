import argparse
import json
from pathlib import Path

from dit_opt.genotype import Genotype
from dit_opt.runtime.evaluator import ProxyEvaluator, ProxyEvaluatorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate final DiT policy genotype.")
    parser.add_argument("--genotype", type=str, required=True, help="Path to serialized genotype (JSON).")
    parser.add_argument("--output", type=str, default="eval_metrics.json", help="Output path for metrics JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    genotype_path = Path(args.genotype)
    genotype_payload = json.loads(genotype_path.read_text())
    genotype = Genotype.from_dict(genotype_payload)

    evaluator = ProxyEvaluator(ProxyEvaluatorConfig())
    metrics = evaluator.evaluate(genotype)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(metrics.__dict__, indent=2))
    print(f"Wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
