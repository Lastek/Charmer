using System;
using System.Linq;
using System.Threading.Tasks;
using Windows.Devices.Enumeration;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;

class Program
{
    static async Task Main()
    {
        // Request camera access (may prompt the user)
        var access = await MediaCapture.RequestAccessAsync();
        if (access != MediaCaptureAccessStatus.Allowed)
        {
            Console.WriteLine("Camera access denied.");
            return;
        }

        // Enumerate all video capture devices
        var devices = await DeviceInformation.FindAllAsync(DeviceClass.VideoCapture);
        Console.WriteLine($"Found {devices.Count} video devices:");

        foreach (var device in devices)
        {
            Console.WriteLine($"  {device.Name} - {device.Id}");
        }

        // Search for a device that contains an infrared source
        MediaCapture mediaCapture = null;
        MediaFrameSource infraredSource = null;

        foreach (var device in devices)
        {
            var group = await MediaFrameSourceGroup.FromIdAsync(device.Id);
            if (group == null) continue;

            var tempMediaCapture = new MediaCapture();
            var settings = new MediaCaptureInitializationSettings
            {
                SourceGroup = group,
                SharingMode = MediaCaptureSharingMode.ExclusiveControl, // Important for secure devices
                MemoryPreference = MediaCaptureMemoryPreference.Cpu
            };

            try
            {
                await tempMediaCapture.InitializeAsync(settings);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to initialize device {device.Name}: {ex.Message}");
                continue;
            }

            // Look for an infrared source
            var irCandidate = tempMediaCapture.FrameSources.FirstOrDefault(
                x => x.Value.Info.SourceKind == MediaFrameSourceKind.Infrared).Value;

            if (irCandidate != null)
            {
                mediaCapture = tempMediaCapture;
                infraredSource = irCandidate;
                Console.WriteLine($"Found infrared source on device: {device.Name}");
                break;
            }
            else
            {
                tempMediaCapture.Dispose();
            }
        }

        if (infraredSource == null)
        {
            Console.WriteLine("No infrared source found. Exiting.");
            return;
        }

        // Create a frame reader for the infrared source
        var frameReader = await mediaCapture.CreateFrameReaderAsync(infraredSource);
        frameReader.FrameArrived += (sender, args) =>
        {
            // Correct: use sender (the frameReader) to acquire the latest frame
            using (var frame = sender.TryAcquireLatestFrame())
            {
                if (frame != null)
                {
                    var videoFrame = frame.VideoMediaFrame?.GetVideoFrame();
                    if (videoFrame != null)
                    {
                        // You can process the frame here (e.g., save or display)
                        Console.WriteLine($"IR frame received at {DateTime.Now:T}");
                    }
                }
            }
        };

        await frameReader.StartAsync();
        Console.WriteLine("Capturing IR frames. Press any key to stop...");
        Console.ReadKey();
        await frameReader.StopAsync();

        mediaCapture.Dispose();
    }
}