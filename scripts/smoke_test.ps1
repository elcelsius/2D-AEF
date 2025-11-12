# scripts/smoke_test.ps1
# Smoke test mínimo: gera preds.csv sintético e valida plots do CIC.
# Extra: força backend "Agg" (headless) e imprime diagnóstico se algo falhar.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# >>> Headless matplotlib
$env:MPLBACKEND = "Agg"
# (opcional) silencia warnings para logs mais limpos
$env:PYTHONWARNINGS = "ignore"

Write-Host "=== Smoke: ambiente ==="
python --version
pip --version

Write-Host "=== Smoke: garantir CLI instalado ==="
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
$csvPath = "outputs\eval_cic\preds.csv"
$csv | Out-File -FilePath $csvPath -Encoding utf8

Write-Host "=== Smoke: rodar plot-eval (fallback em outputs\\eval_cic\\preds.csv) ==="
plot-eval `
  --label_col label `
  --out_dir reports\cic `
  --dataset_tag cic

if ($LASTEXITCODE -ne 0) {
  Write-Host "plot-eval retornou código $LASTEXITCODE"
  Write-Host "Tree de reports após plot-eval:"
  Get-ChildItem -Recurse reports | ForEach-Object { $_.FullName }
  Write-Error "Smoke FAIL: plot-eval retornou erro."
  exit 1
}

Write-Host "=== Smoke: validar artefatos (CIC) ==="
$cic_cm = "reports\cic\confusion_matrix_cic.png"
$cic_f1 = "reports\cic\f1_per_class_cic.png"
$metrics = "reports\cic\metrics_again.json"

$ok1 = Test-Path -Path $cic_cm -PathType Leaf -ErrorAction SilentlyContinue
$ok2 = Test-Path -Path $cic_f1 -PathType Leaf -ErrorAction SilentlyContinue

if (-not ($ok1 -and $ok2)) {
  Write-Host "`n=== DIAGNÓSTICO ==="
  if (Test-Path -Path $metrics) {
    Write-Host "metrics_again.json encontrado. Conteúdo:"
    Get-Content $metrics | Write-Host
  } else {
    Write-Host "metrics_again.json não encontrado."
  }
  Write-Host "`nÁrvore de reports:"
  Get-ChildItem -Recurse reports | ForEach-Object { $_.FullName }
  Write-Error "Smoke FAIL: plots do CIC ausentes."
  exit 1
}

Write-Host "=== Smoke: OK — plots gerados em reports\\cic ==="
exit 0
