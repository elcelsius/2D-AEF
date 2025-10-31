from __future__ import annotations
from pathlib import Path
from typing import Dict

def load_feature_costs(path: str | Path | None = None) -> Dict[str, float]:
    """
    Lê um arquivo simples opcional (CSV ou TXT) com duas colunas:
    feature, cost
    Se não for fornecido, retorna custos vazios => custo padrão 1.0 será usado.
    """
    costs: Dict[str, float] = {}
    if path is None:
        return costs
    p = Path(path)
    if not p.exists():
        return costs
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [x.strip() for x in s.replace(";", ",").split(",")]
        if len(parts) < 2:
            continue
        feat, cost = parts[0], parts[1]
        try:
            costs[feat] = float(cost)
        except ValueError:
            continue
    return costs

def estimate_set_cost(features: list[str], costs: Dict[str, float] | None = None) -> float:
    if not features:
        return 0.0
    costs = costs or {}
    return float(sum(costs.get(f, 1.0) for f in features))
