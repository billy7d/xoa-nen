using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;

internal static class Program
{
    private const string AppTitle = "Local POD Cutout Editor";

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimitInformation
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public long Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimitInformation
    {
        public BasicLimitInformation BasicLimitInformation;
        public IoCounters IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    private const int JobObjectExtendedLimitInformation = 9;
    private const uint JobObjectLimitKillOnJobClose = 0x00002000;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(
        IntPtr job,
        int informationClass,
        IntPtr information,
        uint informationLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr handle);

    [STAThread]
    private static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);

        // Chỉ cho phép một launcher chạy để tránh hai dev server dùng cùng cổng 1420.
        using (var mutex = new Mutex(true, "LocalPODCutoutEditor.WindowsLauncher", out var ownsMutex))
        {
            if (!ownsMutex)
            {
                MessageBox.Show("Ứng dụng đang chạy.", AppTitle, MessageBoxButtons.OK, MessageBoxIcon.Information);
                return 2;
            }

            try
            {
                var projectDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
                var installedApp = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "Local POD Cutout Editor",
                    "local-pod-cutout-editor.exe");
                var npm = @"D:\nodeJS\npm.cmd";
                var python = Path.Combine(projectDir, ".venv", "Scripts", "python.exe");
                var cargoHome = @"E:\DevTools\Rust\cargo";
                var rustupHome = @"E:\DevTools\Rust\rustup";
                var cargoBin = Path.Combine(cargoHome, "bin");
                var runDev = Array.Exists(args, argument => string.Equals(argument, "--dev", StringComparison.OrdinalIgnoreCase));

                // Chế độ kiểm tra dùng khi build launcher, không khởi động giao diện.
                if (args.Length > 0 && string.Equals(args[0], "--check", StringComparison.OrdinalIgnoreCase))
                {
                    if (!File.Exists(installedApp))
                    {
                        ValidateRequiredFiles(projectDir, npm, python, cargoBin);
                    }
                    return 0;
                }

                // Mặc định mở bản đã cài: khởi động nhanh, không dựng Vite/Cargo và không tạo terminal dev.
                if (!runDev && File.Exists(installedApp))
                {
                    return RunInstalledApp(installedApp);
                }

                ValidateRequiredFiles(projectDir, npm, python, cargoBin);
                return RunApp(projectDir, npm, python, cargoHome, rustupHome, cargoBin);
            }
            catch (Exception error)
            {
                MessageBox.Show(error.Message, AppTitle + " - Không thể khởi động", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
        }
    }

    private static void ValidateRequiredFiles(string projectDir, string npm, string python, string cargoBin)
    {
        var requiredFiles = new[]
        {
            Path.Combine(projectDir, "package.json"),
            npm,
            python,
            Path.Combine(cargoBin, "cargo.exe")
        };

        foreach (var file in requiredFiles)
        {
            if (!File.Exists(file))
            {
                throw new FileNotFoundException("Thiếu thành phần cần để chạy app:\n" + file);
            }
        }
    }

    private static int RunApp(
        string projectDir,
        string npm,
        string python,
        string cargoHome,
        string rustupHome,
        string cargoBin)
    {
        var job = CreateKillOnCloseJob();
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = Environment.GetEnvironmentVariable("ComSpec") ?? @"C:\Windows\System32\cmd.exe",
                Arguments = "/d /s /c \"\"" + npm + "\" run tauri dev\"",
                WorkingDirectory = projectDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden
            };

            // Các biến này chỉ áp dụng cho app và tiến trình con, không làm bẩn phiên Windows khác.
            startInfo.EnvironmentVariables["CUTOUT_PYTHON"] = python;
            startInfo.EnvironmentVariables["CUTOUT_BUILD_PYTHON"] = python;
            startInfo.EnvironmentVariables["CARGO_HOME"] = cargoHome;
            startInfo.EnvironmentVariables["RUSTUP_HOME"] = rustupHome;
            startInfo.EnvironmentVariables["PATH"] = cargoBin + ";" + (startInfo.EnvironmentVariables["PATH"] ?? string.Empty);

            using (var process = Process.Start(startInfo))
            {
                if (process == null)
                {
                    throw new InvalidOperationException("Không tạo được tiến trình chạy app.");
                }

                if (!AssignProcessToJobObject(job, process.Handle))
                {
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "Không gắn được tiến trình app vào nhóm giám sát.");
                }

                process.WaitForExit();
                return process.ExitCode;
            }
        }
        finally
        {
            // Đóng Job Object sẽ kết thúc cmd, npm, Vite, Cargo, app và sidecar còn sót.
            CloseHandle(job);
        }
    }

    private static int RunInstalledApp(string executable)
    {
        var job = CreateKillOnCloseJob();
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = executable,
                WorkingDirectory = Path.GetDirectoryName(executable),
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden
            };

            using (var process = Process.Start(startInfo))
            {
                if (process == null)
                {
                    throw new InvalidOperationException("Không tạo được tiến trình bản ứng dụng đã cài.");
                }

                if (!AssignProcessToJobObject(job, process.Handle))
                {
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "Không gắn được ứng dụng đã cài vào nhóm giám sát.");
                }

                process.WaitForExit();
                return process.ExitCode;
            }
        }
        finally
        {
            // Đóng launcher cũng kết thúc đúng tiến trình app đã mở từ launcher.
            CloseHandle(job);
        }
    }

    private static IntPtr CreateKillOnCloseJob()
    {
        var job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error(), "Không tạo được nhóm giám sát tiến trình.");
        }

        var limits = new ExtendedLimitInformation();
        limits.BasicLimitInformation.LimitFlags = JobObjectLimitKillOnJobClose;
        var size = Marshal.SizeOf(typeof(ExtendedLimitInformation));
        var pointer = Marshal.AllocHGlobal(size);

        try
        {
            Marshal.StructureToPtr(limits, pointer, false);
            if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, pointer, (uint)size))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Không bật được chế độ tự dọn tiến trình.");
            }
        }
        catch
        {
            CloseHandle(job);
            throw;
        }
        finally
        {
            Marshal.FreeHGlobal(pointer);
        }

        return job;
    }
}
