# DramaLingo 本地服务一键启动脚本
# 用法：双击 start.bat，或在 PowerShell 中执行 .\start.ps1

# 强制 UTF-8 输出，避免中文乱码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ProjectDir = $PSScriptRoot
Set-Location $ProjectDir

# ── 检查虚拟环境 ──────────────────────────────────────────────
$VenvPython  = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$VenvUvicorn = Join-Path $ProjectDir ".venv\Scripts\uvicorn.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[错误] 未找到虚拟环境，请先执行：" -ForegroundColor Red
    Write-Host "  py -3.11 -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\activate" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    pause
    exit 1
}

if (-not (Test-Path $VenvUvicorn)) {
    Write-Host "[错误] 虚拟环境中未找到 uvicorn，请先执行：" -ForegroundColor Red
    Write-Host "  .venv\Scripts\activate" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    pause
    exit 1
}

# ── 检查 .env 文件 ────────────────────────────────────────────
$EnvFile = Join-Path $ProjectDir ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Host "[警告] 未找到 .env 文件，正在从模板创建..." -ForegroundColor Yellow
    Copy-Item (Join-Path $ProjectDir ".env.example") $EnvFile
    Write-Host "[提示] 请编辑 .env 填入 AI_API_KEY 后重新运行此脚本" -ForegroundColor Cyan
    notepad $EnvFile
    exit 0
}

# ── 检查 AI_API_KEY 是否已填写 ───────────────────────────────
$ApiKey = (Get-Content $EnvFile | Select-String "^AI_API_KEY=").ToString() -replace "^AI_API_KEY=", ""
if ($ApiKey -match "^sk-xxx" -or $ApiKey -eq "") {
    Write-Host "[警告] AI_API_KEY 尚未配置，字幕打标功能将不可用" -ForegroundColor Yellow
}

# ── 检查 FFmpeg ───────────────────────────────────────────────
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "[错误] 未找到 ffmpeg，请先安装：winget install Gyan.FFmpeg" -ForegroundColor Red
    pause
    exit 1
}

# ── 检查并释放端口 8000 ──────────────────────────────────────
$occupied = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($occupied) {
    $pid8000 = ($occupied | Select-Object -First 1).OwningProcess
    $procName = (Get-Process -Id $pid8000 -ErrorAction SilentlyContinue).Name
    Write-Host "[提示] 端口 8000 已被进程 $procName (PID $pid8000) 占用，正在终止..." -ForegroundColor Yellow
    Stop-Process -Id $pid8000 -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Write-Host "[提示] 旧进程已终止，继续启动..." -ForegroundColor Green
}

# ── 启动服务 ──────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DramaLingo 本地服务启动中..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  用户端:   http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  管理后台: http://127.0.0.1:8000/admin" -ForegroundColor Green
Write-Host "  API 文档: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "  按 Ctrl+C 停止服务" -ForegroundColor Gray
Write-Host ""

# ── 等待服务就绪后自动打开浏览器 ─────────────────────────────
Start-Job -Name "OpenBrowser" -ScriptBlock {
    $maxRetries = 30
    $retries = 0
    while ($retries -lt $maxRetries) {
        Start-Sleep -Seconds 1
        try {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) {
                Start-Process "http://127.0.0.1:8000"
                Start-Sleep -Milliseconds 500
                Start-Process "http://127.0.0.1:8000/admin"
                break
            }
        } catch {
            $retries++
        }
    }
} | Out-Null

# 用虚拟环境中的 uvicorn 绝对路径启动，避免 PATH 未生效问题
Write-Host "[诊断] uvicorn 路径: $VenvUvicorn" -ForegroundColor Gray
Write-Host "[诊断] 工作目录: $ProjectDir" -ForegroundColor Gray
Write-Host ""
& $VenvUvicorn main:app --host 127.0.0.1 --port 8000 --reload
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[错误] 服务启动失败，退出码: $LASTEXITCODE" -ForegroundColor Red
}

# 服务停止后清理后台任务
Stop-Job -Name "OpenBrowser" -ErrorAction SilentlyContinue
Remove-Job -Name "OpenBrowser" -ErrorAction SilentlyContinue
