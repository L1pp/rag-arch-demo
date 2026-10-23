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
function Get-ProcessOrNull([int]$ProcessId) {
    try { return [Diagnostics.Process]::GetProcessById($ProcessId) }
    catch [System.ArgumentException] { return $null }
}
$OwnerFile = Assert-CachePath $OwnerFile
if ($CleanupOnly) {
    if (-not (Test-Path -LiteralPath $OwnerFile -PathType Leaf)) {
        throw 'The owned Word identity file is missing; cleanup cannot be verified.'
    }
    try { $owner = Get-Content -LiteralPath $OwnerFile -Raw | ConvertFrom-Json }
    catch { throw 'The owned Word identity file is invalid; cleanup cannot be verified.' }
    if ($owner.owned -ne $true -or $owner.pid -lt 1 -or
        [string]$owner.start_ticks -notmatch '^\d+$' -or
        [IO.Path]::GetFileName([string]$owner.executable) -ine 'WINWORD.EXE') {
        throw 'The owned Word identity file is incomplete; cleanup cannot be verified.'
    }
    $process = Get-ProcessOrNull ([int]$owner.pid)
    if ($null -ne $process) {
        $targetAlreadyExited = $false
        try {
            # Force Process to open and retain its native handle before reading
            # identity fields. Kill and WaitForExit then address that same handle.
            $ownedProcessHandle = $process.Handle
            if ($process.ProcessName -ne 'WINWORD' -or
                $process.StartTime.ToUniversalTime().Ticks.ToString() -ne [string]$owner.start_ticks) {
                $targetAlreadyExited = $true
            }
            elseif ([IO.Path]::GetFullPath($process.Path) -ine
                    [IO.Path]::GetFullPath([string]$owner.executable)) {
                throw 'The recorded Word process executable changed; it was left untouched.'
            }
            else {
                $process.Kill()
                if (-not $process.WaitForExit(10000)) {
                    throw 'The owned Word process did not exit after termination.'
                }
            }
        }
        finally {
            if ($null -ne $process) { $process.Dispose() }
        }
        if ($targetAlreadyExited) {
            Write-Output 'DOCX_PAGE_WORD_TARGET_EXITED_PID_REUSED'
            exit 0
        }
    }
    # A second exact identity lookup closes the Stop-Process/Process.WaitForExit
    # race. A reused PID is left untouched; the original start tick is gone.
    $remaining = Get-ProcessOrNull ([int]$owner.pid)
    if ($null -ne $remaining) {
        try {
            $remainingHandle = $remaining.Handle
            if ($remaining.ProcessName -eq 'WINWORD' -and
                $remaining.StartTime.ToUniversalTime().Ticks.ToString() -eq [string]$owner.start_ticks -and
                [IO.Path]::GetFullPath($remaining.Path) -ieq [IO.Path]::GetFullPath([string]$owner.executable)) {
                throw 'The owned Word process is still running after cleanup.'
            }
        }
        finally {
            if ($null -ne $remaining) { $remaining.Dispose() }
        }
    }
    Write-Output 'DOCX_PAGE_WORD_CLEANUP_VERIFIED'
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
$before = @([Diagnostics.Process]::GetProcessesByName('WINWORD') | ForEach-Object {
    $id = $_.Id
    $_.Dispose()
    $id
})
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
    # Bind the COM object to its actual newly-created PID before opening the
    # document. If startup/open hangs, Python can still clean only this instance.
    Write-Output 'DOCX_PAGE_RENDER_STAGE:identify_word'
    [uint32]$wordProcessId = 0
    [void][WordPageWindow]::GetWindowThreadProcessId([IntPtr]$word.Hwnd, [ref]$wordProcessId)
    if ($wordProcessId -eq 0 -or $before -contains [int]$wordProcessId) {
        throw 'Could not establish ownership of an independent Word instance.'
    }
    $process = Get-ProcessOrNull ([int]$wordProcessId)
    if ($null -eq $process) { throw 'The new Word process exited before its identity could be recorded.' }
    if ($process.ProcessName -ne 'WINWORD' -or [string]::IsNullOrWhiteSpace($process.Path)) {
        throw 'The Word application window does not belong to a verifiable Word process.'
    }
    $wordStartTicks = $process.StartTime.ToUniversalTime().Ticks.ToString()
    $wordExecutable = [IO.Path]::GetFullPath($process.Path)
    if (Test-Path -LiteralPath $OwnerFile) {
        throw 'The Word identity file already exists; refusing to overwrite it.'
    }
    $owned = $true
    $ownerJson = @{ pid = [int]$wordProcessId; owned = $true; start_ticks = $wordStartTicks
        executable = $wordExecutable } |
        ConvertTo-Json -Compress
    $ownerTemp = $OwnerFile + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
    [IO.File]::WriteAllText($ownerTemp, $ownerJson, [Text.UTF8Encoding]::new($false))
    [IO.File]::Move($ownerTemp, $OwnerFile)
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
    if ($owned) {
        $process = Get-ProcessOrNull ([int]$wordProcessId)
        if ($null -ne $process) {
            try {
                $ownedProcessHandle = $process.Handle
                if ($process.ProcessName -eq 'WINWORD' -and
                    $process.StartTime.ToUniversalTime().Ticks.ToString() -eq $wordStartTicks -and
                    [IO.Path]::GetFullPath($process.Path) -ieq $wordExecutable) {
                    if (-not $process.WaitForExit(10000)) {
                        throw 'The owned Word process did not exit after Quit.'
                    }
                }
            }
            finally {
                if ($null -ne $process) { $process.Dispose() }
            }
        }
        $remaining = Get-ProcessOrNull ([int]$wordProcessId)
        if ($null -ne $remaining) {
            try {
                $remainingHandle = $remaining.Handle
                if ($remaining.ProcessName -eq 'WINWORD' -and
                    $remaining.StartTime.ToUniversalTime().Ticks.ToString() -eq $wordStartTicks -and
                    [IO.Path]::GetFullPath($remaining.Path) -ieq $wordExecutable) {
                    throw 'The owned Word process is still running after Quit.'
                }
            }
            finally {
                if ($null -ne $remaining) { $remaining.Dispose() }
            }
        }
    }
}
if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
    throw 'Word did not produce a PDF.'
}
Write-Output 'DOCX_PAGE_RENDER_OK'
