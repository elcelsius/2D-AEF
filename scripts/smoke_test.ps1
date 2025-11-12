# scripts/smoke_test.ps1
# Smoke test sem datasets: cria um preds.csv mínimo e checa geração de plots.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== Smoke: ambiente ==="
python --version
pip --version

Write-Host "=== Smoke: garantir comandos ==="
gatekeeper-train --help | Out-Null
train-specialists --help | Out-Null
infer-twostage --help | Out-Null
eval-twostage --help | Out-Null
plot-eval --help | Out-Null
explain-specialist --help | Out-Null
aggregate-xai --help | Out-Null

Write-Host "=== Smoke: preparar estrutura mínima ==="
New-Item -ItemType Directory -Force -Path outputs\eval_cic | Out-Null
New-Item -ItemType Directory -Force -Path reports\cic | Out-Null
New-Item -ItemType Directory -Force -Path reports\UNSW | Out-Null

Write-Host "=== Smoke: criar preds.csv sintético (binário) ==="
$csv = @"
label,pred_final
0,0
1,1
0,0
1,1
"@
$csv | Out-File -FilePath outputs\eval_cic\preds.csv -Encoding utf8

# Opcional: repetir para UNSW se quiser um CSV no mesmo local de fallback
Copy-Item outputs\eval_cic\preds.csv -Destination reports\UNSW\preds.csv -Force

Write-Host "=== Smoke: gerar plots CIC (usa fallback outputs\\eval_cic\\preds.csv) ==="
plot-eval `
  --label_col label `
  --out_dir reports\cic `
  --dataset_tag cic

Write-Host "=== Smoke: gerar plots UNSW (usa fallback reports\\UNSW\\preds.csv) ==="
plot-eval `
  --label_col label `
  --out_dir reports\UNSW `
  --dataset_tag unsw

Write-Host "=== Smoke: validar artefatos ==="
$cic1 = Test-Path reports\cic\confusion_matrix_cic.png
$cic2 = Test-Path reports\cic\f1_per_class_cic.png
$u1   = Test-Path reports\UNSW\confusion_matrix_unsw.png
$u2   = Test-Path reports\UNSW\f1_per_class_unsw.png

if (-not ($cic1 -and $cic2 -and $u1 -and $u2)) {
  Get-ChildItem -Recurse reports | Write-Host
  Write-Error "Smoke FAIL: plots ausentes (CIC/UNSW)."
}

Write-Host "=== Smoke: OK ==="
exit 0
