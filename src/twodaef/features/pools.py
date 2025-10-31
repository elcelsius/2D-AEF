from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import math
import random

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from twodaef.features.costs import load_feature_costs, estimate_set_cost

@dataclass
class PoolConfig:
    target_col: str
    max_features_per_set: int = 20
    total_sets: int = 30
    seed: int = 42
    # proporção por família
    ratio_pso: float = 0.34
    ratio_gwo: float = 0.33
    ratio_ffa: float = 0.33
    feature_costs_path: str | None = None

def _score_features_mi(df: pd.DataFrame, target_col: str) -> pd.Series:
    y_raw = df[target_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    # --- blacklist para evitar vazamentos ---
    blacklist = {target_col, "Label", "label", "attack_cat", "Attack_cat", "id", "ID"}

    X = df.drop(columns=[c for c in blacklist if c in df.columns])
    # mantém apenas colunas numéricas (evita one-hot agora; simples e rápido)
    X = X.select_dtypes(include=[np.number]).copy()
    # remove colunas com NaN/Inf
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    sel_cols = X.columns.tolist()
    if not sel_cols:
        raise ValueError("Nenhuma coluna numérica disponível após limpeza.")
    mi = mutual_info_classif(X[sel_cols], y, discrete_features=False, random_state=0)
    return pd.Series(mi, index=sel_cols).sort_values(ascending=False)

def _clip_set(s: List[str], k: int) -> List[str]:
    # limita tamanho e remove duplicatas preservando ordem
    seen = set()
    out = []
    for f in s:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
        if len(out) >= k:
            break
    return out

def _random_subset(candidates: List[str], k: int) -> List[str]:
    if k <= 0:
        return []
    if len(candidates) <= k:
        return candidates[:]
    return random.sample(candidates, k)

def _local_mutation(feats: List[str], universe: List[str], k: int, p_swap: float = 0.3) -> List[str]:
    feats = feats[:]
    for i in range(len(feats)):
        if random.random() < p_swap:
            # troca por uma feature aleatória que não esteja no conjunto
            cand = random.choice(universe)
            tries = 0
            while cand in feats and tries < 5:
                cand = random.choice(universe)
                tries += 1
            feats[i] = cand
    return _clip_set(feats, k)

def _fitness_mi(feats: List[str], mi_rank: pd.Series) -> float:
    # fitness simples = soma da MI normalizada das features no set
    if not feats:
        return 0.0
    s = 0.0
    denom = float(mi_rank.max()) if mi_rank.max() > 0 else 1.0
    for f in feats:
        if f in mi_rank.index:
            s += float(mi_rank[f]) / denom
    return s

def _make_pso_like(mi_rank: pd.Series, k: int, n_sets: int, seed: int) -> List[List[str]]:
    random.seed(seed + 17)
    universe = mi_rank.index.tolist()
    sets: List[List[str]] = []
    # partícula inicial: top-k
    gbest = universe[:k]
    sets.append(gbest)
    # iterações curtas para gerar diversidade
    for _ in range(n_sets - 1):
        # "inércia": mantem parte do gbest e muta o resto
        keep = random.randint(max(1, k // 3), max(2, k // 2))
        candidate = _clip_set(gbest[:keep] + _random_subset(universe, k - keep), k)
        # pequena mutação
        candidate = _local_mutation(candidate, universe, k, p_swap=0.2)
        # se novo conjunto tem melhor fitness, atualiza gbest
        if _fitness_mi(candidate, mi_rank) > _fitness_mi(gbest, mi_rank):
            gbest = candidate
        sets.append(candidate)
    return sets

def _make_gwo_like(mi_rank: pd.Series, k: int, n_sets: int, seed: int) -> List[List[str]]:
    random.seed(seed + 31)
    universe = mi_rank.index.tolist()
    sets: List[List[str]] = []
    # alpha/beta/delta iniciais: top-k, mid-k, random-k
    alpha = universe[:k]
    mid_start = max(0, len(universe)//3)
    beta = universe[mid_start:mid_start + k]
    delta = _random_subset(universe, k)
    sets.extend([alpha, beta, delta])
    while len(sets) < n_sets:
        # recombinar alpha/beta/delta com pesos
        w = [0.5, 0.3, 0.2]
        pool = []
        for src, weight in zip([alpha, beta, delta], w):
            take = max(1, int(math.ceil(weight * k)))
            pool.extend(src[:take])
        candidate = _clip_set(pool + _random_subset(universe, k), k)
        candidate = _local_mutation(candidate, universe, k, p_swap=0.25)
        # atualizar alpha/beta/delta se candidatos forem melhores
        if _fitness_mi(candidate, mi_rank) > _fitness_mi(alpha, mi_rank):
            alpha, beta, delta = candidate, alpha, beta
        elif _fitness_mi(candidate, mi_rank) > _fitness_mi(beta, mi_rank):
            beta, delta = candidate, beta
        else:
            delta = candidate
        sets.append(candidate)
    return sets[:n_sets]

def _make_ffa_like(mi_rank: pd.Series, k: int, n_sets: int, seed: int) -> List[List[str]]:
    random.seed(seed + 59)
    universe = mi_rank.index.tolist()
    # inicializa com alguns conjuntos aleatórios + top-k
    population = [universe[:k]] + [_random_subset(universe, k) for _ in range(max(2, n_sets // 3))]
    def brightness(s: List[str]) -> float:
        return _fitness_mi(s, mi_rank)
    while len(population) < n_sets:
        # escolhe dois, “atrai” o pior pelo melhor
        a, b = random.sample(population, 2)
        best, worst = (a, b) if brightness(a) >= brightness(b) else (b, a)
        # atrai: troca algumas features do pior por features do melhor
        merged = _clip_set(worst[:], k)
        for i in range(max(1, k // 4)):
            if i < len(best) and i < len(merged):
                merged[i] = best[i]
        merged = _local_mutation(merged, universe, k, p_swap=0.15)
        population.append(merged)
    return population[:n_sets]

def build_feature_pool(
    df: pd.DataFrame,
    cfg: PoolConfig
) -> Dict[str, Any]:
    random.seed(cfg.seed)
    mi_rank = _score_features_mi(df, cfg.target_col)

    n_pso = max(1, int(round(cfg.total_sets * cfg.ratio_pso)))
    n_gwo = max(1, int(round(cfg.total_sets * cfg.ratio_gwo)))
    n_ffa = max(1, int(cfg.total_sets - n_pso - n_gwo))

    k = cfg.max_features_per_set
    pso_sets = _make_pso_like(mi_rank, k, n_pso, cfg.seed)
    gwo_sets = _make_gwo_like(mi_rank, k, n_gwo, cfg.seed)
    ffa_sets = _make_ffa_like(mi_rank, k, n_ffa, cfg.seed)

    # custo estimado
    costs_map = load_feature_costs(cfg.feature_costs_path)
    def pack(name: str, feats: List[str]) -> Dict[str, Any]:
        return {
            "name": name,
            "features": feats,
            "k": len(feats),
            "score_mi_sum": float(sum(mi_rank.get(f, 0.0) for f in feats)),
            "est_cost": float(estimate_set_cost(feats, costs_map))
        }

    pool = []
    for i, s in enumerate(pso_sets):
        pool.append(pack(f"PSO_{i+1}", s))
    for i, s in enumerate(gwo_sets):
        pool.append(pack(f"GWO_{i+1}", s))
    for i, s in enumerate(ffa_sets):
        pool.append(pack(f"FFA_{i+1}", s))

    # ordena por score_mi_sum decrescente (só para inspeção; manteremos todos)
    pool = sorted(pool, key=lambda d: d["score_mi_sum"], reverse=True)

    return {
        "target_col": cfg.target_col,
        "max_features_per_set": cfg.max_features_per_set,
        "total_sets": cfg.total_sets,
        "seed": cfg.seed,
        "pool": pool,
        "mi_top10": mi_rank.head(10).to_dict(),
        "n_features_universe": int(mi_rank.shape[0]),
    }
