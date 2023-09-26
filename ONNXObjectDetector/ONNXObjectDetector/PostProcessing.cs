using DataTypes;
using Functions;
using ImageONNX;
using ConstantValues;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PostProcessing
{
    public class PostProcessingTool
    {
        private readonly Tensor<float> outputResults;
        private readonly Image<Rgb24> image;
        private readonly double confidance;
        private readonly double iouThreshold;
        public PostProcessingTool(Tensor<float> outputResults, Image<Rgb24> image, double confidance, double iouThreshold)
        {
            this.outputResults = outputResults;
            this.image = image;
            this.confidance = confidance;
            this.iouThreshold = iouThreshold;
        }
        public List<DetectedObject> Run(string imagesOutputDir)
        {
            List<DetectedObject> detectedObjects = new();

            foreach (DetectedObject obj in GetCells())
                if (obj.Confidence > confidance)
                    detectedObjects.Add(obj);

            List<DetectedObject> filteredObjects = ApplyNMS(detectedObjects);

            foreach (DetectedObject obj in filteredObjects)
                image.CropAndSave(obj, imagesOutputDir);

            return filteredObjects;
        }
        private List<DetectedObject> ApplyNMS(List<DetectedObject> detectedObjects)
        {
            List<DetectedObject> filteredObjects = new();
            foreach (DetectedObject obj1 in detectedObjects.OrderByDescending(obj => obj.Confidence).ThenBy(obj => obj.Area))
            {
                bool isOverlapping = false;
                foreach (DetectedObject obj2 in filteredObjects)
                    if ((double)(obj1 & obj2) / (obj1 ^ obj2) > iouThreshold)
                        isOverlapping = true;

                if (!isOverlapping)
                    filteredObjects.Add(obj1);
            }

            return filteredObjects;
        }
        private IEnumerable<DetectedObject> GetCells()
        {
            for (int cy = 0; cy < Constants.CellCount; cy++)
                for (int cx = 0; cx < Constants.CellCount; cx++)
                    for (int b = 0; b < 5; b++)
                        yield return ParseOutputCell(cy, cx, b);
        }
        private (int X, int Y, int W, int H) ConvertDetectedObjectBoxToRealSize(
            float topLeftXONNX, float topLeftYONNX, float widthONNX, float heightONNX)
        {
            double sizeCoeff = (double)Math.Max(image.Height, image.Width) / Constants.TargetSize;
            double paddingX = 0, paddingY = 0;
            if (image.Height >= image.Width)
                paddingX = 0.5 * Constants.TargetSize * (1 - (double)image.Width / image.Height);
            else
                paddingY = 0.5 * Constants.TargetSize * (1 - (double)image.Height / image.Width);

            int X = (int)Math.Max(0, (topLeftXONNX - paddingX) * sizeCoeff);
            int Y = (int)Math.Max(0, (topLeftYONNX - paddingY) * sizeCoeff);
            int W = (int)Math.Min(image.Width - X, widthONNX * sizeCoeff);
            int H = (int)Math.Min(image.Height - Y, heightONNX * sizeCoeff);

            return (X, Y, W, H);
        }
        private static (string detectedClass, float bestClassScore) GetTopClass(float[] lablesProbabilities)
        {
            float[] softValues = FuncTools.SoftMAX(lablesProbabilities);
            int maxValueIndex = 0;
            for (int i = 0; i < softValues.Length; i++)
                if (softValues[i] > softValues[maxValueIndex])
                    maxValueIndex = i;

            return (Constants.Classes[maxValueIndex], softValues[maxValueIndex]);
        }
        private DetectedObject ParseOutputCell(int cx, int cy, int b)
        {
            int offset = b * Constants.ChannelsStep;
            float tx = outputResults[0, offset + Constants.XChannelInd, cy, cx];
            float ty = outputResults[0, offset + Constants.YChannelInd, cy, cx];
            float tw = outputResults[0, offset + Constants.WidthChannelInd, cy, cx];
            float th = outputResults[0, offset + Constants.HeightChannelInd, cy, cx];
            float tc = outputResults[0, offset + Constants.ConfidenceChannelInd, cy, cx];
            float[] tl = Constants.ClassChannelInds
                .Select(channel => outputResults[0, offset + channel, cy, cx])
                .ToArray();

            float width = (float)(Math.Exp(tw) * Constants.YoloAnchors[b].With * Constants.CellSize);
            float height = (float)(Math.Exp(th) * Constants.YoloAnchors[b].Height * Constants.CellSize);

            float topLeftX = (cx + FuncTools.Sigmoid(tx)) * Constants.CellSize - width / 2;
            float topLeftY = (cy + FuncTools.Sigmoid(ty)) * Constants.CellSize - height / 2;

            var (X, Y, W, H) = ConvertDetectedObjectBoxToRealSize(topLeftX, topLeftY, width, height);

            var (detectedClass, bestClassScore) = GetTopClass(tl);
            double confidence = Math.Round(FuncTools.Sigmoid(tc) * bestClassScore * 100, 2);

            return new DetectedObject(
                objClass: detectedClass,
                conf: confidence,
                topLeftX: X,
                topLeftY: Y,
                width: W,
                height: H
            );
        }
    }
}
