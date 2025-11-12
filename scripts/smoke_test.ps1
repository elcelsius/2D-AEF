# scripts/smoke_test.ps1
# Smoke test mínimo: gera um preds.csv sintético (binário) e confere plots do CIC.
# Mantemos UNSW fora do smoke do CI para ser rápido e 100% reprodutível.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== Smoke: ambiente ==="
python --version
pip --version

Write-Host "=== Smoke: garantir CLI instalados ==="
# Se o workflow já roda 'pip install -e .', estes --help apenas validam a presença
plot-eval --help | Out-Null

Write-Host "=== Smoke: preparar estrutura mínima ==="
New-Item -ItemType Directory -Force -Path outputs\eval_cic | Out-Null
New-Item -ItemType Directory -Force -Path reports\cic | Out-Null

Write-Host "=== Smoke: criar preds.csv sintético (binário) ==="
$csv = @"
label,pred_final
0,0
1,1
0,0
1,1
"@
$csv | Out-File -FilePath outputs\eval_cic\preds.csv -Encoding utf8

Write-Host "=== Smoke: gerar plots CIC (usa fallback outputs\\eval_cic\\preds.csv) ==="
plot-eval `
  --label_col label `
  --out_dir reports\cic `
  --dataset_tag cic

Write-Host "=== Smoke: validar artefatos (CIC) ==="
$cic_cm = "reports\cic\confusion_matrix_cic.png"
$cic_f1 = "reports\cic\f1_per_class_cic.png"

$ok1 = Test-Path -Path $cic_cm -PathType Leaf -ErrorAction SilentlyContinue
$ok2 = Test-Path -Path $cic_f1 -PathType Leaf -ErrorAction SilentlyContinue

if (-not ($ok1 -and $ok2)) {
  Get-ChildItem -Recurse reports | Write-Host
  Write-Error "Smoke FAIL: plots do CIC ausentes."
  exit 1
}

Write-Host "=== Smoke: OK — plots gerados em reports\\cic ==="
exit 0
