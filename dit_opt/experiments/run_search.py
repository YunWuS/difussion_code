import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from dit_opt.genotype import Genotype, random_genotype
from dit_opt.runtime.evaluator import ProxyEvaluator, ProxyEvaluatorConfig
from dit_opt.search.ea import EvolutionConfig, EvolutionarySearch
from dit_opt.search.nsga2 import NSGA2Config, NSGA2Search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evolutionary search for DiT policies.")
    parser.add_argument("--config", type=str, required=True, help="Path to search configuration (JSON).")
    parser.add_argument("--output", type=str, default="search_results.json", help="Output path for results JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())

    eval_config = ProxyEvaluatorConfig(
        base_quality=float(config.get("evaluator", {}).get("base_quality", 1.0)),
        base_latency_ms=float(config.get("evaluator", {}).get("base_latency_ms", 100.0)),
        base_mem_mb=float(config.get("evaluator", {}).get("base_mem_mb", 1000.0)),
    )
    evaluator = ProxyEvaluator(eval_config)

    genotype_cfg = config.get("genotype", {})
    num_steps = int(genotype_cfg.get("num_steps", 50))
    num_segments = int(genotype_cfg.get("num_segments", 4))
    attn_scales = genotype_cfg.get("attn_scales", [0.0, 0.25, 0.5, 0.75, 1.0])
    mlp_scales = genotype_cfg.get("mlp_scales", [0.0, 0.25, 0.5, 0.75, 1.0])
    gate_scales = genotype_cfg.get("gate_scales", [0.5, 0.75, 1.0, 1.25])
    cond_gains = genotype_cfg.get("cond_gains", [0.5, 1.0, 1.5])

    search_cfg = config.get("search", {})
    search_type = search_cfg.get("type", "ea")
    population_size = int(search_cfg.get("population_size", 10))
    generations = int(search_cfg.get("generations", 5))
    seed = int(search_cfg.get("seed", 0))

    rng = random.Random(seed)
    initial_population: List[Genotype] = [
        random_genotype(
            num_steps=num_steps,
            num_segments=num_segments,
            attn_scales=attn_scales,
            mlp_scales=mlp_scales,
            gate_scales=gate_scales,
            cond_gains=cond_gains,
            rng=rng,
        )
        for _ in range(population_size)
    ]

    if search_type == "nsga2":
        nsga2_config = NSGA2Config(
            population_size=population_size,
            generations=generations,
            mutation_rate=float(search_cfg.get("mutation_rate", 0.3)),
            seed=seed,
            attn_scales=attn_scales,
            mlp_scales=mlp_scales,
            gate_scales=gate_scales,
            cond_gains=cond_gains,
        )
        searcher = NSGA2Search(evaluator.evaluate)
        final_population = searcher.run(initial_population, nsga2_config)
    else:
        evo_config = EvolutionConfig(
            population_size=population_size,
            generations=generations,
            mutation_rate=float(search_cfg.get("mutation_rate", 0.3)),
            elite_fraction=float(search_cfg.get("elite_fraction", 0.1)),
            quality_threshold=float(search_cfg.get("quality_threshold", 0.0)),
            objective=search_cfg.get("objective", "latency"),
            seed=seed,
            attn_scales=attn_scales,
            mlp_scales=mlp_scales,
            gate_scales=gate_scales,
            cond_gains=cond_gains,
        )
        searcher = EvolutionarySearch(evaluator.evaluate)
        final_population = searcher.run(initial_population, evo_config)

    results: List[Dict[str, Any]] = []
    for genotype in final_population:
        metrics = evaluator.evaluate(genotype)
        results.append(
            {
                "genotype": genotype.to_dict(),
                "metrics": metrics.__dict__,
            }
        )

    output_path = Path(args.output)
    output_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
