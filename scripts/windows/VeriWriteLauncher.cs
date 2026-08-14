using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;

[assembly: AssemblyTitle("VeriWrite Agent Launcher")]
[assembly: AssemblyDescription("VeriWrite Agent MVP Windows launcher")]
[assembly: AssemblyCompany("VeriWrite")]
[assembly: AssemblyProduct("VeriWrite Agent MVP")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

internal static class VeriWriteLauncher
{
    private const string AppUrl = "http://localhost:8501/";
    private const int Port = 8501;

    [STAThread]
    private static int Main(string[] args)
    {
        bool skipBrowser = HasArgument(args, "--no-browser");
        bool reuseServer = HasArgument(args, "--reuse-server");
        string projectDirectory = ResolveProjectDirectory();
        if (projectDirectory == null)
        {
            ShowError(
                "没有找到 VeriWrite 项目目录。\n\n" +
                "请把启动器保留在项目根目录，或设置 VERIWRITE_PROJECT_DIR 环境变量。\n\n" +
                "如果项目已经移动，请重新运行 scripts\\windows\\build_launcher.ps1。",
                "无法启动 VeriWrite Agent");
            return 1;
        }
        string pythonPath = Path.Combine(projectDirectory, ".venv", "Scripts", "python.exe");
        string appPath = Path.Combine(projectDirectory, "streamlit_app.py");

        if (!File.Exists(pythonPath))
        {
            ShowError(
                "没有找到项目虚拟环境：\n\n" + pythonPath +
                "\n\n请将本启动器放在 VeriWrite 项目根目录；若位置正确，请先创建或修复 .venv。",
                "无法启动 VeriWrite Agent");
            return 2;
        }

        if (!File.Exists(appPath))
        {
            ShowError(
                "没有找到 streamlit_app.py。\n\n请将“VeriWrite Agent.exe”保留在项目根目录后重试。",
                "无法启动 VeriWrite Agent");
            return 3;
        }

        EndpointState state = ProbeEndpoint();
        if (state == EndpointState.Ready)
        {
            if (reuseServer)
            {
                return OpenBrowserUnlessSkipped(skipBrowser);
            }
            if (!StopExistingProjectServer(projectDirectory))
            {
                ShowError(
                    "检测到 8501 上已有服务，但无法确认并重启本项目的 Streamlit 进程。\n\n" +
                    "为避免继续显示旧代码，启动器已经停止操作。请关闭原服务后重试。",
                    "无法刷新 VeriWrite Agent 服务");
                return 4;
            }
            state = ProbeEndpoint();
        }

        if (!reuseServer && state == EndpointState.NotRunning)
        {
            // A terminated Streamlit child can leave its Windows venv launcher
            // process behind even though the port is already free. Clean those
            // exact-project processes before creating the replacement server.
            StopExistingProjectServer(projectDirectory);
            state = ProbeEndpoint();
        }

        if (state == EndpointState.PortOccupied)
        {
            // Streamlit may already be starting. Give it a short grace period before failing.
            if (WaitForReady(null, 15000))
            {
                return OpenBrowserUnlessSkipped(skipBrowser);
            }

            ShowError(
                "端口 8501 已被其他程序占用，但没有检测到可用的 VeriWrite 页面。\n\n" +
                "请关闭占用 8501 端口的程序后再次双击启动。",
                "VeriWrite Agent 端口冲突");
            return 4;
        }

        Process streamlitProcess;
        try
        {
            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = pythonPath;
            startInfo.Arguments = "-m streamlit run streamlit_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false";
            startInfo.WorkingDirectory = projectDirectory;
            startInfo.UseShellExecute = false;
            startInfo.CreateNoWindow = true;
            startInfo.WindowStyle = ProcessWindowStyle.Hidden;
            streamlitProcess = Process.Start(startInfo);
        }
        catch (Exception ex)
        {
            ShowError(
                "启动 Streamlit 失败。\n\n" + ex.Message +
                "\n\n可在项目目录运行以下命令排查：\n" +
                ".\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py --server.headless true",
                "无法启动 VeriWrite Agent");
            return 5;
        }

        if (!WaitForReady(streamlitProcess, 90000))
        {
            string detail = streamlitProcess != null && streamlitProcess.HasExited
                ? "Streamlit 进程已退出，退出码：" + streamlitProcess.ExitCode
                : "等待 90 秒后服务仍未就绪。";
            ShowError(
                detail +
                "\n\n请在项目目录运行以下命令查看完整错误：\n" +
                ".\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py --server.headless true",
                "VeriWrite Agent 启动失败");
            return 6;
        }

        return OpenBrowserUnlessSkipped(skipBrowser);
    }

    private static bool HasArgument(string[] args, string expected)
    {
        if (args == null)
        {
            return false;
        }

        foreach (string value in args)
        {
            if (string.Equals(value, expected, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static string ResolveProjectDirectory()
    {
        string executableDirectory = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(
            Path.DirectorySeparatorChar,
            Path.AltDirectorySeparatorChar);
        string configuredDirectory = Environment.GetEnvironmentVariable("VERIWRITE_PROJECT_DIR");
        string[] candidates = new string[]
        {
            executableDirectory,
            configuredDirectory,
            Environment.CurrentDirectory
        };

        foreach (string candidate in candidates)
        {
            if (string.IsNullOrWhiteSpace(candidate))
            {
                continue;
            }

            string fullPath;
            try
            {
                fullPath = Path.GetFullPath(candidate);
            }
            catch
            {
                continue;
            }

            if (File.Exists(Path.Combine(fullPath, "streamlit_app.py")) &&
                File.Exists(Path.Combine(fullPath, ".venv", "Scripts", "python.exe")))
            {
                return fullPath;
            }
        }

        return null;
    }

    private static bool WaitForReady(Process process, int timeoutMilliseconds)
    {
        Stopwatch stopwatch = Stopwatch.StartNew();
        while (stopwatch.ElapsedMilliseconds < timeoutMilliseconds)
        {
            if (ProbeEndpoint() == EndpointState.Ready)
            {
                return true;
            }

            if (process != null && process.HasExited)
            {
                return false;
            }

            Thread.Sleep(500);
        }

        return false;
    }

    private static EndpointState ProbeEndpoint()
    {
        try
        {
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create(AppUrl);
            request.Method = "GET";
            request.Timeout = 1500;
            request.ReadWriteTimeout = 1500;
            request.AllowAutoRedirect = true;
            request.Proxy = null;
            request.UserAgent = "VeriWrite-Launcher/1.0";

            using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
            {
                if ((int)response.StatusCode >= 200 && (int)response.StatusCode < 400)
                {
                    return EndpointState.Ready;
                }
            }
        }
        catch
        {
            // A closed port and a service that is still booting both reach this branch.
        }

        return IsPortOccupied() ? EndpointState.PortOccupied : EndpointState.NotRunning;
    }

    private static bool IsPortOccupied()
    {
        TcpClient client = new TcpClient();
        try
        {
            IAsyncResult result = client.BeginConnect("127.0.0.1", Port, null, null);
            bool connected = result.AsyncWaitHandle.WaitOne(500);
            if (!connected)
            {
                return false;
            }

            client.EndConnect(result);
            return client.Connected;
        }
        catch
        {
            return false;
        }
        finally
        {
            client.Close();
        }
    }

    private static bool StopExistingProjectServer(string projectDirectory)
    {
        string escapedProject = projectDirectory.Replace("'", "''");
        string command =
            "$project='" + escapedProject + "';" +
            "$processes=Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | " +
            "Where-Object {$_.Name -eq 'python.exe' " +
            "-and $_.CommandLine -like ('*'+$project+'*') " +
            "-and $_.CommandLine -like '*streamlit*streamlit_app.py*'};" +
            "foreach($target in $processes){" +
            "Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue" +
            "}";
        try
        {
            ProcessStartInfo stopInfo = new ProcessStartInfo();
            stopInfo.FileName = "powershell.exe";
            stopInfo.Arguments =
                "-NoProfile -NonInteractive -WindowStyle Hidden -Command \"" +
                command.Replace("\"", "\\\"") + "\"";
            stopInfo.UseShellExecute = false;
            stopInfo.CreateNoWindow = true;
            stopInfo.WindowStyle = ProcessWindowStyle.Hidden;
            Process stopProcess = Process.Start(stopInfo);
            if (stopProcess == null || !stopProcess.WaitForExit(15000) || stopProcess.ExitCode != 0)
            {
                return false;
            }
        }
        catch
        {
            return false;
        }

        Stopwatch stopwatch = Stopwatch.StartNew();
        while (stopwatch.ElapsedMilliseconds < 10000)
        {
            if (!IsPortOccupied())
            {
                return true;
            }
            Thread.Sleep(250);
        }
        return false;
    }

    private static int OpenBrowserUnlessSkipped(bool skipBrowser)
    {
        if (skipBrowser)
        {
            return 0;
        }

        try
        {
            ProcessStartInfo browser = new ProcessStartInfo();
            browser.FileName = AppUrl;
            browser.UseShellExecute = true;
            Process.Start(browser);
            return 0;
        }
        catch (Exception ex)
        {
            ShowError(
                "系统已经启动，但无法自动打开浏览器。\n\n请手动访问：" + AppUrl +
                "\n\n详细信息：" + ex.Message,
                "VeriWrite Agent 已启动");
            return 7;
        }
    }

    private static void ShowError(string message, string title)
    {
        MessageBox.Show(message, title, MessageBoxButtons.OK, MessageBoxIcon.Error);
    }

    private enum EndpointState
    {
        NotRunning,
        PortOccupied,
        Ready
    }
}
