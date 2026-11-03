# Final test-set evaluation of the trained checkpoints, most important results first.
# Safe to run again: benchmark.py skips every condition already in its CSV.
Set-Location (Split-Path $PSScriptRoot -Parent | Split-Path -Parent)
Remove-Item camo\results\evaluation_done -ErrorAction SilentlyContinue

function Run([string]$out, [string]$data, [string[]]$models, [string[]]$options) {
    & .venv\Scripts\python.exe camo\scripts\benchmark.py --models $models --data camo\data\$data `
        --out camo\results\test_$out.csv @options *>> camo\results\evaluate.log
}

$cod = "COD-TestDataset\COD10K"
$all8 = "--corruptions", "gaussian_noise", "gaussian_blur", "motion_blur", "brightness",
        "contrast", "fog", "jpeg", "rain", "--severities", "3", "--combined"
$hard = "--corruptions", "motion_blur", "gaussian_noise", "jpeg", "--severities", "5"
$input_only = "--protocols", "input"
$native = "--protocols", "native", "--corruptions", "motion_blur", "gaussian_noise", "jpeg",
          "--severities", "3", "5", "--combined"

function Input([string]$out, [string]$data, [string[]]$models) {
    Run $out $data $models ($input_only + $all8)
    Run $out $data $models ($input_only + $hard)
}

# 1. The main comparison on COD10K.
Input "cod10k" $cod @("arm_E", "arm_B", "sam2unet")

# 2. Real footage, as it was filmed.
Run "moca" "MoCA_Video" @("sam2unet", "arm_B", "arm_C", "arm_E") ($input_only + "--corruptions")

# 3. Images with no camouflaged object. Positives come from the COD10K run above.
Run "absent" "COD10K-v3\Test" @("sam2unet", "arm_E") ($input_only + "--corruptions" + "--combined")

# 4. The remaining arms and the older detector.
Input "cod10k" $cod @("arm_C", "arm_D", "arm_R", "sinet")

# 5. Why the gate does what it does, and the quality of the real footage.
& .venv\Scripts\python.exe camo\scripts\gate_analysis.py *>> camo\results\evaluate.log

# 6. Damage before resizing, for the protocol study.
Run "cod10k" $cod @("sam2unet", "sinet", "arm_B", "arm_E") $native

# 7. Secondary test sets.
Input "camo" "COD-TestDataset\CAMO" @("arm_E", "arm_B", "sam2unet", "arm_C", "arm_D", "arm_R")
Run "nc4k" "COD-TestDataset\NC4K" @("sam2unet", "arm_B", "arm_E") ($input_only + "--corruptions" + "--combined")

New-Item camo\results\evaluation_done -ItemType File | Out-Null
