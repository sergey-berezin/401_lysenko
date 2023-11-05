using ConstantValues;
using DataTypes;
using Functions;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.Formats.Jpeg;

namespace ImageONNX
{
    public static class ImageONNXExtensions
    {
        public static DenseTensor<float> GetTensorForONNX(this Image<Rgb24> image)
        {
            DenseTensor<float> input = new(new[] { 1, 3, Constants.TargetSize, Constants.TargetSize });

            using Image<Rgb24> resized = image.Clone(x => x.Resize(
                new ResizeOptions
                {
                    Size = new Size(Constants.TargetSize, Constants.TargetSize),
                    Mode = ResizeMode.Pad
                }
            ));
            resized.ProcessPixelRows(pa =>
            {
                for (int y = 0; y < Constants.TargetSize; y++)
                {
                    Span<Rgb24> pixelSpan = pa.GetRowSpan(y);
                    for (int x = 0; x < Constants.TargetSize; x++)
                    {
                        input[0, 0, y, x] = pixelSpan[x].R;
                        input[0, 1, y, x] = pixelSpan[x].G;
                        input[0, 2, y, x] = pixelSpan[x].B;
                    }
                }
            });

            return input;
        }
        public static void CropAndSave(this Image<Rgb24> image, DetectedObject obj, string pathToSave)
        {
            string outputFileName = $"{FuncTools.GenerateImageId(5)}.jpg";
            string outputFilePath = $"{pathToSave}\\{outputFileName}";
            var (X, Y, W, H) = obj.BoxProperties;

            using Image<Rgb24> croppedImage = image.Clone(x => x.Crop(new Rectangle(X, Y, W, H)));
            croppedImage.Save(outputFilePath, new JpegEncoder());

            obj.OutputFileName = Path.GetFileName(outputFileName);
        }
    }
}
