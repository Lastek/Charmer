#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <Mferror.h>
#include <wrl/client.h>
#include <vector>
#include <string>
#include <iostream>

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

using Microsoft::WRL::ComPtr;

std::vector<std::wstring> EnumerateCaptureDevices() {
    std::vector<std::wstring> deviceNames;
    UINT32 count = 0;
    IMFAttributes* pAttributes = nullptr;
    IMFActivate** ppDevices = nullptr;

    HRESULT hr = MFCreateAttributes(&pAttributes, 1);
    if (SUCCEEDED(hr)) {
        hr = pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    }
    if (SUCCEEDED(hr)) {
        hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);
    }

    for (UINT32 i = 0; i < count; ++i) {
        LPWSTR name = nullptr;
        UINT32 nameLen = 0;
        hr = ppDevices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &name, &nameLen);
        if (SUCCEEDED(hr)) {
            deviceNames.push_back(std::wstring(name, nameLen));
            CoTaskMemFree(name);
        }
        ppDevices[i]->Release();
    }
    CoTaskMemFree(ppDevices);
    if (pAttributes) pAttributes->Release();
    return deviceNames;
}

int wmain() {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) return 1;

    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        CoUninitialize();
        return 1;
    }

    auto devices = EnumerateCaptureDevices();

    if (devices.empty()) {
        std::wcout << L"No video capture devices found." << std::endl;
    } else {
        for (size_t i = 0; i < devices.size(); ++i) {
            std::wcout << L"[" << i << L"] " << devices[i] << std::endl;
        }
    }

    MFShutdown();
    CoUninitialize();
    return 0;
}
