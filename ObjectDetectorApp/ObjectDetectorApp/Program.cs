using ONNXObjectDetector;

namespace ObjectDetectorApp
{
    class Program
    {
        static ObjectDetectionService objectDetectionService = new();
        static SemaphoreSlim outputFileLock = new(1, 1);
        static async Task Main(string[] args)
        {
            if (!File.Exists("results.csv"))
                File.WriteAllLines("results.csv", new string[] { string.Join(",", new string[] { "filename", "class", "x", "y", "w", "h" }) });

            try { objectDetectionService.InitONNXModelSession(); }
            catch (Exception e)
            {
                Console.WriteLine(e.Message, e.StackTrace);
                return;
            }

            using var tokenSource = new CancellationTokenSource();
            var tasks = args.Select(imagePath => objectDetectionService.RunAsync(imagePath, tokenSource.Token)).ToList();
            var image_paths = args.ToList();
            while (true)
            {
                if (!tasks.Any())
                    break;

                try
                {
                    var completed_task = await Task.WhenAny(tasks);
                    string task_image_path = image_paths[tasks.IndexOf(completed_task)];
                    tasks.Remove(completed_task);
                    image_paths.Remove(task_image_path);

                    await outputFileLock.WaitAsync();
                    File.AppendAllLines("results.csv", completed_task.Result.Select(obj => obj.ToCSVRow()));
                    Console.WriteLine($"Successful {task_image_path}");

                    outputFileLock.Release();
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message, e.StackTrace);
                    outputFileLock.Release();
                }
            }
        }
    }
}
