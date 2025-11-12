# scripts/smoke_test.ps1
# Smoke test mínimo: valida que o pacote instala e que geramos plots do CIC.
# (UNSW fica fora do smoke para manter o job leve e reprodutível no CI.)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "== 2D-AEF :: Smoke Test =="
python --version
pip --version

Write-Host "== install package (editable) =="
pip install -e .

Write-Host "== generate CIC plots =="
# Usa o CLI que já sabe resolver automaticamente o preds.csv (dataset_tag=cic)
# e grava os PNGs com sufixo _cic em reports\cic\.
plot-eval `
  --label_col label `
  --out_dir reports\cic `
  --dataset_tag cic

# Verificação dos arquivos (apenas CIC)
$cic_cm = "reports\cic\confusion_matrix_cic.png"
$cic_f1 = "reports\cic\f1_per_class_cic.png"

if (Test-Path $cic_cm -PathType Leaf -ErrorAction SilentlyContinue -and
    Test-Path $cic_f1 -PathType Leaf -ErrorAction SilentlyContinue) {
  Write-Host "Smoke OK — plots em reports\cic"
} else {
  Write-Error "Smoke FAIL: plots do CIC ausentes."
}
