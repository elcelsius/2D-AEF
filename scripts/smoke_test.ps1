# scripts/smoke_test.ps1
$ErrorActionPreference = "Stop"
Write-Host "== 2D-AEF Smoke Test =="

# 1) Mini dados (CIC binário já que é rápido)
python scripts/prep_cic_train.py

# 2) Gatekeeper rápido
gatekeeper-train `
  --train_csv data\train_cic.csv `
  --target_col label `
  --features gatekeeper_cic_cols.txt `
  --model_out artifacts\gatekeeper_cic.joblib

# 3) Especialistas (um conjunto rápido)
train-specialists `
  --train_csv data\train_cic.csv `
  --target_col label `
  --feature_pool_json artifacts\feature_pool_cic.json `
  --out_dir artifacts\specialists_cic `
  --map_path artifacts\specialist_map_cic.json `
  --models lgbm `
  --max_features_per_set 20

# 4) Avaliação + plots
eval-twostage `
  --gatekeeper_model artifacts\gatekeeper_cic.joblib `
  --gatekeeper_features gatekeeper_cic_cols.txt `
  --specialist_map artifacts\specialist_map_cic.json `
  --input_csv data\cic_eval.csv `
  --label_col label `
  --output_dir outputs\eval_cic

plot-eval `
  --label_col label `
  --out_dir reports\cic `
  --dataset_tag cic

# 5) Verificações mínimas
if (!(Test-Path "reports\cic\confusion_matrix_cic.png") -or !(Test-Path "reports\cic\f1_per_class_cic.png")) {
  Write-Error "Smoke FAIL: PNGs não foram gerados."
  exit 1
}
if (!(Test-Path "outputs\eval_cic\preds.csv")) {
  Write-Error "Smoke FAIL: outputs\eval_cic\preds.csv ausente."
  exit 1
}

Write-Host "SMOKE OK"
exit 0
