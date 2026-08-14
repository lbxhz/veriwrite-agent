$ErrorActionPreference = "Stop"

$projectDirectory = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$sourcePath = Join-Path $PSScriptRoot "VeriWriteLauncher.cs"
$outputPath = Join-Path $projectDirectory "VeriWrite Agent.exe"
$desktopDirectory = [Environment]::GetFolderPath("Desktop")
$desktopShortcutPath = Join-Path $desktopDirectory "VeriWrite Agent MVP.lnk"
$compilerPath = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"

if (-not (Test-Path -LiteralPath $compilerPath)) {
    throw "未找到 Windows C# 编译器：$compilerPath"
}

& $compilerPath `
    /nologo `
    /target:winexe `
    /platform:anycpu `
    /optimize+ `
    "/out:$outputPath" `
    /reference:System.dll `
    /reference:System.Windows.Forms.dll `
    $sourcePath

if ($LASTEXITCODE -ne 0) {
    throw "启动器编译失败，退出码：$LASTEXITCODE"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($desktopShortcutPath)
$shortcut.TargetPath = $outputPath
$shortcut.WorkingDirectory = $projectDirectory
$shortcut.Description = "启动 VeriWrite Agent MVP"
$shortcut.Save()

Write-Host "启动器已生成：$outputPath"
Write-Host "桌面快捷方式已生成：$desktopShortcutPath"
