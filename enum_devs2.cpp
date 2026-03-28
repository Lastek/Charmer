#define _WIN32_WINNT _WIN32_WINNT_WIN10   // or _WIN32_WINNT_WIN8, etc.
// #define _WIN32_WINNT _WIN32_WINNT_WIN7
#define INITGUID  // Ensure GUIDs are defined as constants (optional, but helps)

#include <windows.h>
#include <setupapi.h>
#include <devguid.h>
#include <ks.h>
#include <ksmedia.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#ifndef MF_E_DEVICE_IN_USE
#define MF_E_DEVICE_IN_USE 0xC00D36E4L
#endif
// DirectShow headers
#include <dshow.h>
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "strmiids.lib")

// For printing GUIDs
std::string GuidToString(const GUID& guid) {
    wchar_t guidStr[40];
    StringFromGUID2(guid, guidStr, 40);
    char buf[40];
    WideCharToMultiByte(CP_UTF8, 0, guidStr, -1, buf, 40, NULL, NULL);
    return std::string(buf);
}

// Helper to print last error
void PrintLastError(const char* msg) {
    DWORD err = GetLastError();
    std::cerr << msg << " error: " << err << std::endl;
}

// Enumerate device paths for a given interface class GUID (wide version)
std::vector<std::wstring> EnumerateDevicePaths(const GUID& interfaceClass) {
    std::vector<std::wstring> paths;
    std::cout << "Enumerating GUID: " << GuidToString(interfaceClass) << std::endl;

    HDEVINFO devInfoSet = SetupDiGetClassDevs(&interfaceClass, NULL, NULL, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    if (devInfoSet == INVALID_HANDLE_VALUE) {
        PrintLastError("SetupDiGetClassDevs");
        return paths;
    }

    SP_DEVICE_INTERFACE_DATA devInterfaceData = { sizeof(SP_DEVICE_INTERFACE_DATA) };
    DWORD index = 0;
    while (SetupDiEnumDeviceInterfaces(devInfoSet, NULL, &interfaceClass, index, &devInterfaceData)) {
        DWORD requiredSize = 0;
        SetupDiGetDeviceInterfaceDetail(devInfoSet, &devInterfaceData, NULL, 0, &requiredSize, NULL);
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            PrintLastError("SetupDiGetDeviceInterfaceDetail (size query)");
            index++;
            continue;
        }

        PSP_DEVICE_INTERFACE_DETAIL_DATA_W detailData = (PSP_DEVICE_INTERFACE_DETAIL_DATA_W)malloc(requiredSize);
        detailData->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W);

        if (SetupDiGetDeviceInterfaceDetailW(devInfoSet, &devInterfaceData, detailData, requiredSize, NULL, NULL)) {
            paths.push_back(detailData->DevicePath);
            std::wcout << L"  Found path: " << detailData->DevicePath << std::endl;
        } else {
            PrintLastError("SetupDiGetDeviceInterfaceDetailW (data retrieval)");
        }
        free(detailData);
        index++;
    }
    DWORD err = GetLastError();
    if (err != ERROR_NO_MORE_ITEMS) {
        std::cerr << "SetupDiEnumDeviceInterfaces stopped with error: " << err << std::endl;
    }
    SetupDiDestroyDeviceInfoList(devInfoSet);
    return paths;
}

// DirectShow enumeration
void EnumerateDirectShowDevices() {
    std::cout << "\n--- DirectShow Devices ---\n";
    ICreateDevEnum* pDevEnum = NULL;
    IEnumMoniker* pEnum = NULL;
    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER,
                                   IID_ICreateDevEnum, (void**)&pDevEnum);
    if (SUCCEEDED(hr)) {
        hr = pDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &pEnum, 0);
        if (hr == S_OK) {
            IMoniker* pMoniker = NULL;
            while (pEnum->Next(1, &pMoniker, NULL) == S_OK) {
                IPropertyBag* pPropBag;
                hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag, (void**)&pPropBag);
                if (SUCCEEDED(hr)) {
                    VARIANT var;
                    VariantInit(&var);
                    hr = pPropBag->Read(L"FriendlyName", &var, 0);
                    if (SUCCEEDED(hr)) {
                        std::wcout << L"  DirectShow device: " << var.bstrVal << std::endl;
                        VariantClear(&var);
                    }
                    // Try to get device path
                    hr = pPropBag->Read(L"DevicePath", &var, 0);
                    if (SUCCEEDED(hr)) {
                        std::wcout << L"    Path: " << var.bstrVal << std::endl;
                        VariantClear(&var);
                    }
                    pPropBag->Release();
                }
                pMoniker->Release();
            }
            pEnum->Release();
        } else {
            std::cout << "  No DirectShow video devices found (hr=" << std::hex << hr << std::dec << ")\n";
        }
        pDevEnum->Release();
    } else {
        std::cout << "  Failed to create device enumerator (hr=" << std::hex << hr << std::dec << ")\n";
    }
}

// Try to create an IMFMediaSource from a device path
HRESULT CreateMediaSourceFromPath(const std::wstring& path, IMFMediaSource** ppSource) {
    *ppSource = nullptr;
    IMFAttributes* pAttributes = nullptr;
    HRESULT hr = MFCreateAttributes(&pAttributes, 2);
    if (SUCCEEDED(hr)) {
        hr = pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    }
    if (SUCCEEDED(hr)) {
        hr = pAttributes->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, path.c_str());
    }
    if (SUCCEEDED(hr)) {
        hr = MFCreateDeviceSource(pAttributes, ppSource);
    }
    if (pAttributes) pAttributes->Release();
    return hr;
}

int main() {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (SUCCEEDED(hr)) {
        hr = MFStartup(MF_VERSION);
    }
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize Media Foundation.\n";
        return 1;
    }

    // Enumerate both KSCATEGORY_VIDEO_CAMERA and KSCATEGORY_VIDEO
    std::vector<std::wstring> paths = EnumerateDevicePaths(KSCATEGORY_VIDEO_CAMERA);
    auto videoPaths = EnumerateDevicePaths(KSCATEGORY_VIDEO);
    paths.insert(paths.end(), videoPaths.begin(), videoPaths.end());

    std::cout << "\nTotal unique device paths found: " << paths.size() << std::endl;

    // If no paths, try DirectShow as fallback
    if (paths.empty()) {
        EnumerateDirectShowDevices();
    }

    for (const auto& path : paths) {
        std::wcout << L"\nDevice path: " << path << std::endl;

        IMFMediaSource* pSource = nullptr;
        hr = CreateMediaSourceFromPath(path, &pSource);
        if (SUCCEEDED(hr)) {
            std::cout << "  Successfully created media source.\n";

            IMFSourceReader* pReader = nullptr;
            hr = MFCreateSourceReaderFromMediaSource(pSource, nullptr, &pReader);
            if (SUCCEEDED(hr)) {
                // Check multiple streams
                for (DWORD streamIndex = 0; ; streamIndex++) {
                    DWORD typeIndex = 0;
                    while (true) {
                        IMFMediaType* pType = nullptr;
                        hr = pReader->GetNativeMediaType(streamIndex, typeIndex, &pType);
                        if (hr == MF_E_NO_MORE_TYPES) break;
                        if (hr == MF_E_INVALIDSTREAMNUMBER) goto next_stream;
                        if (SUCCEEDED(hr)) {
                            GUID subtype = GUID_NULL;
                            pType->GetGUID(MF_MT_SUBTYPE, &subtype);
                            UINT32 width = 0, height = 0;
                            MFGetAttributeSize(pType, MF_MT_FRAME_SIZE, &width, &height);
                            UINT32 num = 0, denom = 0;
                            MFGetAttributeRatio(pType, MF_MT_FRAME_RATE, &num, &denom);
                            float fps = (denom > 0) ? (float)num / denom : 0.0f;

                            std::string subtypeStr = GuidToString(subtype);
                            std::cout << "    Stream " << streamIndex << ", type " << typeIndex
                                      << ": " << subtypeStr << " " << width << "x" << height
                                      << " " << fps << " fps" << std::endl;
                            pType->Release();
                            typeIndex++;
                        }
                    }
                }
                next_stream:
                pReader->Release();
            } else {
                std::cout << "  Failed to create source reader (hr=0x" << std::hex << hr << std::dec << ")\n";
            }
            pSource->Release();
        } else {
            std::cout << "  Failed to create media source (hr=0x" << std::hex << hr << std::dec << ")\n";
            if (hr == MF_E_DEVICE_IN_USE) {
                std::cout << "    Device is in use (likely by Windows Hello).\n";
            }
        }
    }

    MFShutdown();
    CoUninitialize();
    return 0;
}