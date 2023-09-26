using ConstantValues;
using System.Net;

namespace Functions
{
    public class FuncTools
    {
        public static string GenerateImageId(int length = 5) => Guid.NewGuid().ToString()[..length];
        public static float Sigmoid(float x) => 1f / (1f + (float)Math.Exp(-x));
        public static float[] SoftMAX(float[] values)
        {
            IEnumerable<double> exps = values.Select(x => Math.Exp(x));
            double sum = exps.Sum();
            return exps.Select(x => (float)(x / sum)).ToArray();
        }
        public static void PrepareModel()
        {
            if (File.Exists(Constants.ModelName))
                return;

            int retryCount = 0;

            using WebClient webClient = new();
            while (retryCount < Constants.MaxRetries)
            {
                try
                {
                    webClient.DownloadFile(Constants.ModelURL, Constants.ModelName);
                    return;
                }
                catch (Exception)
                {
                    retryCount++;
                }
            }

            throw new Exception("Cannot load ONNX model file");
        }
    }
}
