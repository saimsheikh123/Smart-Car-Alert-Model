# Deploy trained model to Audio_Models folder
# Usage: .\deploy_model.ps1

$SOURCE_CHECKPOINT = "c:\Users\Saim\cmpe-281-models\cmpe-281-models\train\checkpoints\best_audiocrnn_7class.pth"
$SOURCE_FINAL = "c:\Users\Saim\cmpe-281-models\cmpe-281-models\train\checkpoints\multi_audio_crnn.pth"
$DEST_FOLDER = "c:\Users\Saim\cmpe-281-models\cmpe-281-models\Audio_Models\Audio_Models"
$DEST_NAME = "multi_audio_crnn.pth"

Write-Host "[DEPLOY] Starting model deployment..." -ForegroundColor Green

# Check if checkpoint exists
if (Test-Path $SOURCE_CHECKPOINT) {
    Write-Host "✓ Found best checkpoint: $SOURCE_CHECKPOINT"
    
    # Copy checkpoint
    $dest_path = Join-Path $DEST_FOLDER $DEST_NAME
    Copy-Item -Path $SOURCE_CHECKPOINT -Destination $dest_path -Force
    Write-Host "✓ Copied to: $dest_path" -ForegroundColor Green
    
} elseif (Test-Path $SOURCE_FINAL) {
    Write-Host "✓ Found final checkpoint: $SOURCE_FINAL"
    
    # Copy checkpoint
    $dest_path = Join-Path $DEST_FOLDER $DEST_NAME
    Copy-Item -Path $SOURCE_FINAL -Destination $dest_path -Force
    Write-Host "✓ Copied to: $dest_path" -ForegroundColor Green
} else {
    Write-Host "✗ No checkpoint found!" -ForegroundColor Red
    Write-Host "  Expected: $SOURCE_CHECKPOINT or $SOURCE_FINAL"
    exit 1
}

Write-Host "`n[DEPLOY] Model deployment complete!" -ForegroundColor Green
Write-Host "Next step: Restart FastAPI server to load the new model"
Write-Host "Run: cd $DEST_FOLDER; python -m uvicorn multi_model_api:app --host 127.0.0.1 --port 8000"
