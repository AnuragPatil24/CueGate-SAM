# Train the arms one after another, in priority order: B, E, D, then C.
# Safe to run again: each arm resumes from last.pth, and a finished arm exits at once.
Set-Location (Split-Path $PSScriptRoot -Parent | Split-Path -Parent)
Remove-Item camo\results\training_done -ErrorAction SilentlyContinue
foreach ($arm in "B", "E", "D", "C") {
    & .venv\Scripts\python.exe camo\scripts\train.py --arm $arm --epochs 8 --batch 6 *>> camo\results\train.log
}
New-Item camo\results\training_done -ItemType File | Out-Null
