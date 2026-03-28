Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | ? { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
Function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}
[Windows.Devices.Enumeration.DeviceInformation,Windows.System.Devices,ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Capture.Frames.MediaFrameSourceGroup,Windows.Media.Capture,ContentType=WindowsRuntime] | Out-Null

$devices = Await ([Windows.Devices.Enumeration.DeviceInformation]::FindAllAsync([Windows.Devices.Enumeration.DeviceClass]::VideoCapture)) ([System.Collections.Generic.IReadOnlyList[Windows.Devices.Enumeration.DeviceInformation]])
$devices | % {
    Write-Host "Device: $($_.Name)"
    $group = Await ([Windows.Media.Capture.Frames.MediaFrameSourceGroup]::FromIdAsync($_.Id)) ([Windows.Media.Capture.Frames.MediaFrameSourceGroup])
    if ($group -ne $null) {
        $group.SourceInfos | % {
            Write-Host "  Source: $($_.Id), Kind: $($_.SourceKind), StreamType: $($_.MediaStreamType)"
        }
    }
}