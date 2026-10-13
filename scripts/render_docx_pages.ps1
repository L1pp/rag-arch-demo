param(
    [string]$InputPath,
    [string]$OutputPath,
    [Parameter(Mandatory = $true)][string]$OwnerFile,
    [switch]$CleanupOnly
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$cacheRoot = [IO.Path]::GetFullPath((Join-Path (Split-Path $PSScriptRoot -Parent) 'data/runtime/document_pages'))
function Assert-CachePath([string]$Value) {
    $resolved = [IO.Path]::GetFullPath($Value)
    if (-not $resolved.StartsWith($cacheRoot + [IO.Path]::DirectorySeparatorChar,
                                  [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Only the registered document cache may be converted.'
    }
    return $resolved
}
$OwnerFile = Assert-CachePath $OwnerFile
if ($CleanupOnly) {
    if (Test-Path -LiteralPath $OwnerFile) {
        $owner = Get-Content -LiteralPath $OwnerFile -Raw | ConvertFrom-Json
        $process = Get-Process -Id ([int]$owner.pid) -ErrorAction SilentlyContinue
        if ($null -ne $process -and $owner.owned -eq $true -and
            $process.ProcessName -eq 'WINWORD' -and
            $process.StartTime.ToUniversalTime().Ticks.ToString() -eq [string]$owner.start_ticks) {
            Stop-Process -Id $process.Id -Force
        }
    }
    exit 0
}
$InputPath = Assert-CachePath $InputPath
$OutputPath = Assert-CachePath $OutputPath
if ([IO.Path]::GetExtension($InputPath) -ne '.docx' -or
    [IO.Path]::GetExtension($OutputPath) -ne '.pdf' -or
    -not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
    throw 'A cached DOCX input and PDF output are required.'
}
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class WordPageWindow {     [DllImport("user32.dll")]     public static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint processId); }'
$before = @(Get-Process -Name WINWORD -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
$word = $null
$document = $null
$documentWindow = $null
$documents = $null
$options = $null
$owned = $false
$oldOptions = @{}
try {
    # CreateObject starts our own Word session; GetObject is deliberately not used.
    Write-Output 'DOCX_PAGE_RENDER_STAGE:create_word'
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $documents = $word.Documents
    # Open only the validated cache copy, read-only and without a recent-file entry.
    $missing = [Type]::Missing
    $noPassword = '__document_pages_no_password__'
    Write-Output 'DOCX_PAGE_RENDER_STAGE:open_document'
    $document = $documents.Open($InputPath, $false, $true, $false,
        $noPassword, $noPassword, $false, $noPassword, $noPassword,
        0, $missing, $false, $false, $missing, $true)
    # Do not acquire shutdown authority until the document window identifies a new PID.
    Write-Output 'DOCX_PAGE_RENDER_STAGE:identify_window'
    $documentWindow = $document.ActiveWindow
    [uint32]$wordProcessId = 0
    [void][WordPageWindow]::GetWindowThreadProcessId([IntPtr]$documentWindow.Hwnd, [ref]$wordProcessId)
    if ($wordProcessId -eq 0 -or $before -contains [int]$wordProcessId) {
        throw 'Could not establish ownership of an independent Word instance.'
    }
    $process = Get-Process -Id $wordProcessId
    if ($process.ProcessName -ne 'WINWORD') {
        throw 'The document window does not belong to a Word process.'
    }
    $owned = $true
    @{ pid = [int]$wordProcessId; owned = $true
       start_ticks = $process.StartTime.ToUniversalTime().Ticks.ToString() } |
        ConvertTo-Json | Set-Content -LiteralPath $OwnerFile -Encoding UTF8
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $options = $word.Options
    foreach ($name in @('UpdateLinksAtOpen', 'UpdateFieldsAtPrint', 'UpdateLinksAtPrint')) {
        $oldOptions[$name] = $options.$name
        $options.$name = $false
    }
    Write-Output 'DOCX_PAGE_RENDER_STAGE:repaginate'
    $document.Repaginate()
    Write-Output 'DOCX_PAGE_RENDER_STAGE:export_pdf'
    $document.ExportAsFixedFormat($OutputPath, 17)
}
finally {
    if ($owned) {
        if ($null -ne $document) { try { $document.Close(0) } catch {} }
        if ($null -ne $options) {
            foreach ($name in $oldOptions.Keys) { try { $options.$name = $oldOptions[$name] } catch {} }
        }
        if ($null -ne $word) { try { $word.Quit(0) } catch {} }
    }
    foreach ($com in @($document, $documentWindow, $documents, $options, $word)) {
        if ($null -ne $com -and [Runtime.InteropServices.Marshal]::IsComObject($com)) {
            [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($com)
        }
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
    throw 'Word did not produce a PDF.'
}
Write-Output 'DOCX_PAGE_RENDER_OK'
