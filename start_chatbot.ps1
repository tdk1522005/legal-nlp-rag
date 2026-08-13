$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentRoot = Split-Path -Parent $ProjectRoot

$PythonExe = Join-Path `
    $ParentRoot `
    ".venv\Scripts\python.exe"

$WebApp = Join-Path `
    $ProjectRoot `
    "web_app.py"

$LlamaServer = "C:\Users\tdkha\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"

$HostAddress = "127.0.0.1"
$LlamaPort = 8080

function Test-LlamaPort {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect(
            $HostAddress,
            $LlamaPort
        )
        $client.Close()
        return $true
    }
    catch {
        return $false
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  LEGAL NLP RAG - QWEN + STREAMLIT"
Write-Host "=========================================="
Write-Host ""

if (-not (Test-Path $PythonExe)) {
    throw "Không tìm thấy Python trong .venv: $PythonExe"
}

if (-not (Test-Path $WebApp)) {
    throw "Không tìm thấy web_app.py: $WebApp"
}

if (-not (Test-Path $LlamaServer)) {
    throw "Không tìm thấy llama-server.exe: $LlamaServer"
}

$StartedLlamaServer = $false
$LlamaProcess = $null

if (Test-LlamaPort) {
    Write-Host "[OK] Qwen server đang chạy tại port 8080."
}
else {
    Write-Host "[1/2] Đang khởi động Qwen3-4B..."

    $LlamaArguments = @(
        "-hf",
        "ggml-org/Qwen3-4B-GGUF:Q4_K_M",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--jinja"
    )

    $LlamaProcess = Start-Process `
        -FilePath $LlamaServer `
        -ArgumentList $LlamaArguments `
        -WindowStyle Minimized `
        -PassThru

    $StartedLlamaServer = $true

    Write-Host "Đang đợi Qwen load model..."

    $TimeoutSeconds = 180
    $Elapsed = 0

    while (-not (Test-LlamaPort)) {
        Start-Sleep -Seconds 2
        $Elapsed += 2

        if ($LlamaProcess.HasExited) {
            throw "Qwen server đã dừng trong lúc khởi động."
        }

        if ($Elapsed -ge $TimeoutSeconds) {
            Stop-Process `
                -Id $LlamaProcess.Id `
                -Force `
                -ErrorAction SilentlyContinue

            throw "Qwen server khởi động quá thời gian cho phép."
        }

        Write-Host "." -NoNewline
    }

    Write-Host ""
    Write-Host "[OK] Qwen3-4B đã sẵn sàng."
}

Write-Host ""
Write-Host "[2/2] Đang khởi động Streamlit..."
Write-Host ""
Write-Host "Chatbot: http://localhost:8501"
Write-Host ""
Write-Host "Nhấn Ctrl + C để dừng chatbot."
Write-Host ""

try {
    Set-Location $ProjectRoot

    & $PythonExe `
        -m streamlit `
        run `
        $WebApp `
        --server.fileWatcherType `
        none
}
finally {
    if (
        $StartedLlamaServer `
        -and $null -ne $LlamaProcess
    ) {
        Write-Host ""
        Write-Host "Đang tắt Qwen server..."

        Stop-Process `
            -Id $LlamaProcess.Id `
            -Force `
            -ErrorAction SilentlyContinue

        Write-Host "[OK] Đã tắt Qwen server."
    }
}
