namespace DataTypes
{
    public class DetectedObject
    {
        private readonly string @class;
        private readonly double confidence;
        private readonly int x;
        private readonly int y;
        private readonly int w;
        private readonly int h;

        public string OutputFileName = string.Empty;
        public double Confidence => confidence;
        public int Area => h * w;
        public (int X, int Y, int W, int H) BoxProperties => (x, y, w, h);
        public DetectedObject(string objClass, double conf, int topLeftX, int topLeftY, int width, int height)
        {
            @class = objClass;
            confidence = conf;
            x = topLeftX;
            y = topLeftY;
            w = width;
            h = height;
        }
        public static int operator |(DetectedObject obj1, DetectedObject obj2) => obj1.Area + obj2.Area;
        public static int operator &(DetectedObject obj1, DetectedObject obj2)
        {
            bool classesIsEq = obj1.@class == obj2.@class;

            int x1 = Math.Max(obj1.x, obj2.x);
            int y1 = Math.Max(obj1.y, obj2.y);
            int x2 = Math.Min(obj1.x + obj1.w, obj2.x + obj2.w);
            int y2 = Math.Min(obj1.y + obj1.h, obj2.y + obj2.h);

            if (!classesIsEq || x1 >= x2 || y1 >= y2)
                return 0;

            return (x2 - x1) * (y2 - y1);
        }
        public static int operator ^(DetectedObject obj1, DetectedObject obj2) => (obj1 | obj2) - (obj1 & obj2);
        public string ToCSVRow() => string.Join(",", new object[] { OutputFileName, @class, x, y, w, h });
    }
}
