using ONNXObjectDetector;

namespace YOLO_csharp
{
    class Program
    {
        static ObjectDetectionService objectDetectionService = new();
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
            var tasks = args.Select(imagePath => objectDetectionService.RunAsync(imagePath, tokenSource.Token)).ToArray();
            try
            {
                await Task.WhenAll(tasks);
                foreach (var task in tasks)
                    File.AppendAllLines("results.csv", task.Result.Select(obj => obj.ToCSVRow()));
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message, e.StackTrace);
                return;
            }
        }
    }
}
