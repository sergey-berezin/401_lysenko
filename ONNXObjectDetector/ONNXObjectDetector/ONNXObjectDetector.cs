using Microsoft.ML.OnnxRuntime;
using ImageONNX;
using PostProcessing;
using DataTypes;
using Functions;
using ConstantValues;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ONNXObjectDetector
{
    public class ObjectDetectionService
    {
        private InferenceSession? session;
        public double Confidence = Constants.Confidence;
        public double IouThreshold = Constants.IouThreshold;
        public void InitONNXModelSession()
        {
            FuncTools.PrepareModel();
            using SessionOptions opt = new();
            opt.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
            session = new InferenceSession(Constants.ModelName, opt);
        }
        public Task<List<DetectedObject>> RunAsync(string imagePath, CancellationToken token)
        {
            return Task.Factory.StartNew(_ => Run(imagePath), token, TaskCreationOptions.LongRunning);
        }
        public List<DetectedObject> Run(string imagePath)
        {
            if (session is null)
                throw new Exception("Please call this.InitONNXModelSession to init session.");

            Image<Rgb24> image = Image.Load<Rgb24>(imagePath);
            DenseTensor<float> input = image.GetTensorForONNX();
            List<NamedOnnxValue> inputs = new() { NamedOnnxValue.CreateFromTensor("image", input) };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
            lock (session)
                results = session.Run(inputs);

            Tensor<float> output = results.First().AsTensor<float>();
            PostProcessingTool postProcessingTool = new(output, image, Confidence, IouThreshold);

            string outputImagesDir = "cropped_images";
            if (!Directory.Exists(outputImagesDir))
                Directory.CreateDirectory(outputImagesDir);

            List<DetectedObject> detectedObjects = postProcessingTool.Run(outputImagesDir);
            results.Dispose();

            return detectedObjects;
        }
    }
}
