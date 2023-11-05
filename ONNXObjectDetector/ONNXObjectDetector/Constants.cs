namespace ConstantValues
{
    public class Constants
    {
        public const string ModelURL = "https://storage.yandexcloud.net/dotnet4/tinyyolov2-8.onnx";
        public const string ModelName = "tinyyolov2-8.onnx";
        public const int MaxRetries = 5;

        public const int ChannelsStep = 25;
        public const int TargetSize = 416;
        public const int CellCount = 13;
        public const int CellSize = TargetSize / CellCount;
        public static readonly (double With, double Height)[] YoloAnchors = {
            (1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52)
        };

        public const int XChannelInd = 0;
        public const int YChannelInd = 1;
        public const int WidthChannelInd = 2;
        public const int HeightChannelInd = 3;
        public const int ConfidenceChannelInd = 4;
        public const int ClassesCount = 20;
        public static readonly IEnumerable<int> ClassChannelInds = Enumerable.Range(5, ClassesCount);
        public static readonly string[] Classes = {
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        };

        public const double Confidence = 50;
        public const double IouThreshold = 0.6;
    }
}
