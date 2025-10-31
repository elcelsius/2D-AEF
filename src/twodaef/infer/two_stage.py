from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import json
import time
import joblib
import numpy as np
import pandas as pd
from loguru import logger

# Repetições para cronometria robusta (min de várias execuções)
GK_BENCH_REPEATS = 7     # gatekeeper (batch)
S2_BENCH_REPEATS = 3     # especialista (por linha)


@dataclass
class TwoStageConfig:
    gatekeeper_model: str
    gatekeeper_features_file: str  # arquivo texto com uma feature por linha
    specialist_map_json: str       # artifacts/specialist_map.json
    input_csv: str                 # dados para inferência
    output_csv: str                # onde salvar as predições
    fill_missing: float = 0.0      # valor para preencher colunas ausentes
    gatekeeper_labelmap_json: Optional[str] = None  # mapeia saída do GK -> chave do especialista


class TwoStageInferencer:
    def __init__(self, cfg: TwoStageConfig):
        self.cfg = cfg
        self.gatekeeper = self._load_gatekeeper(cfg.gatekeeper_model)
        self.gk_features = self._load_feature_list(cfg.gatekeeper_features_file)
        self.spec_map = self._load_specialists(cfg.specialist_map_json)  # {class_name: {...}}

        # Auto-descoberta de labelmap se não for fornecido
        lm_path: Optional[str] = cfg.gatekeeper_labelmap_json
        if not lm_path:
            # tenta ao lado do specialist_map e depois em artifacts/
            cand = [
                Path(cfg.specialist_map_json).parent / "gk_labelmap_unsw.json",
                Path("artifacts/gk_labelmap_unsw.json"),
                Path("artifacts/gk_labelmap.json"),
            ]
            for p in cand:
                if p.exists():
                    lm_path = str(p)
                    break
        self.labelmap = self._load_labelmap(lm_path)  # pode ser {}

    @staticmethod
    def _load_gatekeeper(path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Gatekeeper não encontrado: {p}")
        return joblib.load(p)

    @staticmethod
    def _load_feature_list(path: str) -> List[str]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo de features do gatekeeper não encontrado: {p}")
        feats: List[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            feats.append(s)
        return feats

    @staticmethod
    def _load_specialists(path: str) -> Dict[str, Any]:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        specs: Dict[str, Any] = {}
        for cls_name, payload in d.get("specialists", {}).items():
            mpath = Path(payload["model_path"])
            if not mpath.exists():
                logger.warning(f"Modelo do especialista ausente para classe {cls_name}: {mpath}")
                continue
            model = joblib.load(mpath)
            feats = list(payload["features"])
            specs[str(cls_name)] = {
                "model": model,
                "features": feats,
                "model_key": payload.get("model_key", ""),
                "feature_set_name": payload.get("feature_set_name", "")
            }
        if not specs:
            raise RuntimeError("Nenhum especialista carregado a partir do mapa.")
        return specs

    @staticmethod
    def _load_labelmap(path: Optional[str]) -> Dict[str, str]:
        if not path:
            return {}
        p = Path(path)
        if not p.exists():
            logger.warning(f"Label map não encontrado: {p}")
            return {}
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            # normaliza chaves/valores para string
            return {str(k): str(v) for k, v in d.items()}
        except Exception as e:
            logger.warning(f"Falha ao ler label map {p}: {e}")
            return {}

    @staticmethod
    def _heuristic_bin_map(x: Any) -> Optional[str]:
        """
        Heurística: se rótulo textual contém 'benign' ou 'normal' => '0', senão => '1'.
        Inteiros permanecem como '0'/'1'.
        """
        try:
            # se é número (ex.: 0/1), normaliza para str e retorna
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            s = str(x).strip().lower()
            if s in {"0", "1"}:
                return s
            # palavras que denotam tráfego benigno
            if any(tok in s for tok in ("benign", "normal", "clean", "legit")):
                return "0"
            return "1"
        except Exception:
            return None

    def _ensure_columns(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for c in cols:
            if c not in df.columns:
                df[c] = self.cfg.fill_missing
        # manter ordem
        return df[cols]

    def predict_csv(self) -> Tuple[float, float, float]:
        # 1) carregar dados
        df = pd.read_csv(self.cfg.input_csv)
        n = df.shape[0]
        if n == 0:
            raise ValueError("input_csv não possui linhas.")

        # 2) etapa 1 — gatekeeper
        Xgk = self._ensure_columns(df, self.gk_features)

        # Alguns modelos salvos do gatekeeper retornam (y_pred, lat_ms) ou (y_pred, meta)
        _res = self.gatekeeper.predict(Xgk)
        if isinstance(_res, tuple):
            gk_pred = _res[0]
        else:
            gk_pred = _res

        # garantir shape 1-D
        gk_pred = np.asarray(gk_pred).ravel()

        # warm-up para estabilizar caches/JIT
        _ = self.gatekeeper.predict(Xgk.iloc[: min(512, len(Xgk))])

        # bench robusto do gatekeeper (batch): melhor tempo / n
        gk_best_ns = None
        for _rep in range(GK_BENCH_REPEATS):
            ns0 = time.perf_counter_ns()
            _ = self.gatekeeper.predict(Xgk)
            ns1 = time.perf_counter_ns()
            dt = ns1 - ns0
            gk_best_ns = dt if (gk_best_ns is None or dt < gk_best_ns) else gk_best_ns
        gk_ms = (gk_best_ns / 1e6) / max(1, n)

        # 3) etapa 2 — especialista por linha
        final_pred: List[int] = []
        spec_used: List[str] = []
        spec_set: List[str] = []
        stage2_times: List[float] = []
        gk_mapped: List[str] = []  # novo: mapa aplicado na saída do GK

        for i, gp in enumerate(gk_pred):
            # 3.1) aplica labelmap (se houver)
            cls: Optional[str] = None
            if self.labelmap:
                key = str(gp)
                mapped = self.labelmap.get(key)
                if mapped is not None:
                    cls = str(mapped)

            # 3.2) se não mapeou, tenta heurística (binário)
            if cls is None:
                cls = self._heuristic_bin_map(gp)

            # 3.3) se ainda None, usa str direto (último recurso)
            if cls is None:
                cls = str(gp)

            gk_mapped.append(cls)  # guarda o mapeado (string '0'/'1' na prática do UNSW)

            spec = self.spec_map.get(cls)
            if spec is None:
                # fallback: coerção segura para 0/1
                try:
                    yhat_int = int(cls)
                except Exception:
                    h = self._heuristic_bin_map(cls)
                    yhat_int = int(h) if h is not None else 0

                final_pred.append(yhat_int)
                spec_used.append("fallback_gk")
                spec_set.append("NA")
                stage2_times.append(0.0)
                continue

            feats = spec["features"]
            model = spec["model"]
            row_df = df.iloc[[i]]  # manter DataFrame
            Xsp = self._ensure_columns(row_df, feats)

            # bench robusto por linha (pega o mínimo de S2_BENCH_REPEATS)
            best_ns = None
            yhat = None
            for _rep in range(S2_BENCH_REPEATS):
                ns2 = time.perf_counter_ns()
                yhat = model.predict(Xsp)[0]
                ns3 = time.perf_counter_ns()
                dt = ns3 - ns2
                best_ns = dt if (best_ns is None or dt < best_ns) else best_ns

            # garante inteiro 0/1 para avaliação numérica
            try:
                yhat_int = int(yhat)
            except Exception:
                # se por algum motivo veio string, aplica heurística
                h = self._heuristic_bin_map(yhat)
                yhat_int = int(h) if h is not None else 0

            final_pred.append(yhat_int)
            spec_used.append(spec.get("model_key", ""))
            spec_set.append(spec.get("feature_set_name", ""))
            stage2_times.append(best_ns / 1e6)  # ms

        # 4) métricas de latência
        stage2_ms = float(np.mean(stage2_times)) if stage2_times else 0.0
        total_ms = gk_ms + stage2_ms

        # 5) salvar CSV de saída
        out = df.copy()
        out["pred_gatekeeper"] = gk_pred
        out["pred_gatekeeper_mapped"] = gk_mapped  # novo: útil p/ depuração e avaliação
        out["pred_final"] = final_pred
        out["specialist_model"] = spec_used
        out["specialist_featureset"] = spec_set
        out["latency_ms_stage1"] = round(gk_ms, 6)
        out["latency_ms_stage2"] = [round(x, 6) for x in stage2_times]
        out["latency_ms_total_est"] = round(total_ms, 6)

        Path(self.cfg.output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(self.cfg.output_csv, index=False)
        logger.success(f"Predições salvas em {self.cfg.output_csv}")
        logger.info(
            f"Latência média — Gatekeeper: {gk_ms:.6f} ms | "
            f"Especialista: {stage2_ms:.6f} ms | Total: {total_ms:.6f} ms/linha"
        )
        return gk_ms, stage2_ms, total_ms
