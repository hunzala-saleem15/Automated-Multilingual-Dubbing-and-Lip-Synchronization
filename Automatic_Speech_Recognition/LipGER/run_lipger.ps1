# ==============================
# Step -1: Activate virtual environment
# ==============================
Write-Host '🔹 Activating virtual environment...'
& 'D:\ASR\LipGER\venv\Scripts\Activate.ps1'

# ==============================
# Step 0: Paths
# ==============================
$train_json = 'D:\ASR\facestar_whisper\facestar_full_train_whisper_fixed.json'
$test_json  = 'D:\ASR\facestar_whisper\facestar_full_test_whisper_fixed.json'

$train_pt = 'D:\ASR\facestar_whisper\facestar_full_train.pt'
$test_pt  = 'D:\ASR\facestar_whisper\facestar_full_test.pt'

$model_checkpoint = 'D:\ASR\lipreading_best.pth'
$output_dir = 'D:\ASR\lipger_finetuned'
$pred_output = 'D:\ASR\facestar_whisper\predictions.json'

# ==============================
# Step 1: Convert JSON -> PT
# ==============================
Write-Host '🔹 Converting TRAIN JSON to PT...'
python 'D:\ASR\LipGER\scripts\convert_to_pt.py' `
    --json_path $train_json `
    --output_path $train_pt

Write-Host '🔹 Converting TEST JSON to PT...'
python 'D:\ASR\LipGER\scripts\convert_to_pt.py' `
    --json_path $test_json `
    --output_path $test_pt

# ==============================
# Step 2: Train / Fine-tune LipGER
# ==============================
Write-Host '🔹 Fine-tuning LipGER model...'
Set-Location 'D:\ASR\LipGER'
python -m lipger.lipger `
    --train_path $train_pt `
    --val_path $train_pt `
    --checkpoint_dir $output_dir `
    --model_checkpoint $model_checkpoint `
    --batch_size 4 `
    --lr 1e-4 `
    --epochs 10 `
    --device cuda

# ==============================
# Step 3: Inference / Test
# ==============================
Write-Host '🔹 Running inference on test set...'
python -m lipger.lipger `
    --test_path $test_pt `
    --checkpoint_dir $output_dir `
    --output_path $pred_output `
    --device cuda

# ==============================
# Step 4: Finish message
# ==============================
Write-Host 'LipGER training & inference completed'
Write-Host 'Predictions saved at:' , $pred_output
