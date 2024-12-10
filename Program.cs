using System;
using System.Xml;
using System.Drawing;
using System.Diagnostics;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;

class Program {
    // compare pixel by pixel to a master image
    static Mat CompareImages(Mat img1, Mat img2) {
        Mat gray1 = new Mat();
        Mat gray2 = new Mat();
        CvInvoke.CvtColor(img2, gray1, ColorConversion.Bgr2Gray);
        CvInvoke.CvtColor(img1, gray2, ColorConversion.Bgr2Gray);

        Mat diff = new Mat();
        CvInvoke.AbsDiff(gray1, gray2, diff);

        Mat thresh = new Mat();
        CvInvoke.Threshold(diff, thresh, 30, 255, ThresholdType.Binary);

        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        Mat hierarchy = new Mat();
        CvInvoke.FindContours(thresh, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

        for (int i = 0; i < contours.Size; i++) {
            Rectangle boundingBox = CvInvoke.BoundingRectangle(contours[i]);
            Point center = new Point(boundingBox.X + boundingBox.Width / 2, boundingBox.Y + boundingBox.Height / 2);
            int radius = Math.Max(boundingBox.Width, boundingBox.Height) / 2;
            CvInvoke.Circle(img2, center, radius, new MCvScalar(0, 255, 0), 2);
        }

        return img2;
    }
    
    static public string MissingHole_getInPathNoExt(string dir, int index) {
        string number = index < 10 ? "0" + index : "" + index;
        string fileNameNoExt = "01_missing_hole_" + number;
        // string fileName = fileNameNoExt + ".jpg";
        string path = dir + fileNameNoExt;
        return path;
    }

    static public string MouseBite_getInPathNoExt(string dir, int index) {
        string number = index < 10 ? "0" + index : "" + index;
        string fileNameNoExt = "01_mouse_bite_" + number;
        string path = dir + fileNameNoExt;
        return path;
    }

    static public string MissingHole_getOutPathNoExt(string dir, int index) {
        string outDirs = dir + index + "/";
        Directory.CreateDirectory(outDirs);
        
        string number = index < 10 ? "0" + index : "" + index;
        string fileNameNoExt = "01_missing_hole_" + number;
        // string outNameColor = fileNameNoExt + "_Color" + ".jpg";
        string outPath = outDirs + fileNameNoExt;
        return outPath;
    }

    static public string MouseBite_getOutPathNoExt(string dir, int index) {
        string outDirs = dir + index + "/";
        Directory.CreateDirectory(outDirs);
        
        string number = index < 10 ? "0" + index : "" + index;
        string fileNameNoExt = "01_mouse_bite_" + number;
        string outPath = outDirs + fileNameNoExt;
        return outPath;
    }

    static void Main(string[] args) {
        string outDir = "output/";
        string missingHoleDir = "PCB_DATASET/images/Missing_hole/";
        string mouseBiteDir = "PCB_DATASET/images/Mouse_bite/";
        if (Directory.Exists(outDir)) Directory.Delete(outDir, true);
        Directory.CreateDirectory(outDir);
        
        bool missingHole = true;
        bool mouseBite = true;
        for(int i = 1; i <= 20; ++i) {
            Console.WriteLine($"Scanning image {i}:");
            if(missingHole) {
                Console.WriteLine($"  Finding missing holes");
                string inPath = MissingHole_getInPathNoExt(missingHoleDir, i) + ".jpg";
                var img = new Image<Bgr, Byte>(inPath);

                var res = DetectMissingHole(img);

                string outPath = MissingHole_getOutPathNoExt(outDir, i) + ".jpg";
                CvInvoke.Imwrite(outPath, res);
            }

            if(mouseBite) {
                Console.WriteLine($"  Finding mouse bites");
                string inPath = MouseBite_getInPathNoExt(mouseBiteDir, i) + ".jpg";
                var img = new Image<Bgr, Byte>(inPath);

                var res = DetectMouseBite(img);

                string outPath = MouseBite_getOutPathNoExt(outDir, i) + ".jpg";
                CvInvoke.Imwrite(outPath, res);
            }
        }
    }

    static Image<Bgr, Byte> DetectMissingHole(Image<Bgr, Byte> img) {
        return
            new ProcessedImage<Bgr>(img)
            .ConvertToGrayscale()
            .SmoothGaussian(5)
            .ThresholdBinary(new Gray(40), new Gray(255))
            .Morphology(MorphOp.Close, ElementShape.Ellipse, new Size(8, 8))
            .ThresholdBinaryInv(new Gray(128), new Gray(255))
            .GetComponents()
            .FilterComponentsBySize(200, 600)
            .SmoothGaussian(5)
            .FindCircles(new Gray(255), new Gray(50), 5, 1, 0, 20)
            // NOTE: iterate over all the circles, iterate the pixels inside the circle,
            //       get the percent of the circle filled with white pixels, if below a threshold discard the circle
            .FindCirclesWithFillPercent(0.4)
            .DrawCircles(img);
    }

    static public Rectangle toRect(CircleF c) {
        int x = (int)c.Center.X - (int)c.Radius;
        int y = (int)c.Center.Y - (int)c.Radius;
        int width = (int)c.Radius * 2;
        int height = (int)c.Radius * 2;
        return new Rectangle(x, y, width, height);
    }

    static public bool isPointInCircle(Point p, CircleF c) {
        PointF center = c.Center;

        var dist = Math.Sqrt(Math.Abs(Math.Pow(p.X - (int)center.X, 2)) + Math.Abs(Math.Pow(p.Y - (int)center.Y, 2)));
        if (dist > c.Radius) return false;
        return true;
    }

    // TODO: instead of using morphology operations to get the lines, insted use the distance field
    //       only color the pixels white that are within a range, meaning they fall on the edge (is this the same this are finding contours?)
    static public Image<Bgr, Byte> DetectMouseBite(Image<Bgr, Byte> img) {
        return
            new ProcessedImage<Bgr>(img)
            .SmoothGaussian(5)
            .InRange(new Bgr(0, 80, 0), new Bgr(40, 100, 45))
            .GetComponents()
            .FilterComponentsBySize(125, 37500)
            .Morphology(MorphOp.Dilate, ElementShape.Rectangle, new Size(3, 3))
            .Morphology(MorphOp.Open, ElementShape.Rectangle, new Size(5, 5))
            .Morphology(MorphOp.Close, ElementShape.Ellipse, new Size(5, 5))
            // running copper lines highlighted
            .GenerateDistanceMask(DistType.L1, 10)
            .FilterByDistance(15, 30) // max: 2295
            // outlines of the wire
            .Morphology(MorphOp.Dilate, ElementShape.Ellipse, new Size(4, 4))
            .Morphology(MorphOp.Open, ElementShape.Ellipse, new Size(3, 3))
            .GenerateDistanceMask(DistType.L1, 10)
            // TODO: use inv version
            .FilterByDistance(34, 2295) // max: 2295
            // only the larger white chunks remain
            .GetComponents()
            .HighlightComponents()
            .DrawCircles(img);
    }
}

class ProcessedImage<TColor>
where TColor : struct, IColor
{
    Image<TColor, Byte> image;
    
    // variables used when finding components
    Mat componentsLabels = new();
    Mat componentsStats = new();
    Mat componentsCentroids = new();
    int labelsCount;

    // variables used for distance field
    Mat distanceLabels = new();

    // the result of HoughCircles
    public CircleF[][] circles = new CircleF[1][];
    // the result of HoughLines
    LineSegment2D[][] lines = new LineSegment2D[1][];

    public ProcessedImage(string filepath) {
        this.image = new Image<TColor, Byte>(filepath);
    }

    public ProcessedImage(Image<TColor, Byte> img) {
        this.image = img.Clone();
    }

    ProcessedImage(
        Image<TColor, Byte> image,
        Mat componentsLabels,
        Mat componentsStats,
        Mat componentsCentroids,
        int labelsCount,
        Mat distanceLabels,
        CircleF[][] circles,
        LineSegment2D[][] lines
    ) {
        this.image = image;
        this.componentsLabels = componentsLabels;
        this.componentsStats = componentsStats;
        this.componentsCentroids = componentsCentroids;
        this.labelsCount = labelsCount;
        this.distanceLabels = distanceLabels;
        this.circles = circles;
        this.lines = lines;
    }

    public Image<Gray, Byte> ImageToGray() {
        return this.image.Convert<Gray, Byte>();
    }

    public Image<Bgr, Byte> ImageToColor() {
        return this.image.Convert<Bgr, Byte>();
    }

    public ProcessedImage<TColor> SmoothGaussian(int kernelSize) {
        this.image = this.image.SmoothGaussian(kernelSize);
        return this;
    }

    public ProcessedImage<Gray> InRange(TColor bot, TColor top) {
        var gray = this.image.InRange(bot, top);
        return new ProcessedImage<Gray>(
            gray,
            this.componentsLabels,
            this.componentsStats,
            this.componentsCentroids,
            this.labelsCount,
            this.distanceLabels,
            this.circles,
            this.lines
        );
    }

    public ProcessedImage<TColor> GetComponents() {
        this.labelsCount = CvInvoke.ConnectedComponentsWithStats(
            this.image,              // IInputArray image,
            this.componentsLabels,   // IOutputArray labels,
            this.componentsStats,    // IOutputArray stats,
            this.componentsCentroids // IOutputArray centroids,
            // LineType connectivity = LineType.EightConnected,
            // DepthType labelType = DepthType.Cv32S,
            // ConnectedComponentsAlgorithmsTypes cclType = ConnectedComponentsAlgorithmsTypes.Default
        );
        return this;
    }

    public ProcessedImage<TColor> FilterComponentsBySize(int minArea, int maxArea) {
        var iStats = this.componentsStats.ToImage<Gray, Int32>();
        var iLabels = this.componentsLabels.ToImage<Gray, Int32>();
        for (int row = 0; row < this.image.Rows; row++) {
            for (int col = 0; col < this.image.Cols; col++) {
                Int32 componentIdx = iLabels.Data[row, col, 0];
                if (componentIdx == 0) continue;
                Int32 componentArea = iStats.Data[componentIdx, 4, 0];

                if(minArea < componentArea && componentArea < maxArea) {
                    // white
                    this.image.Data[row, col, 0] = (Byte)255;
                } else {
                    // black
                    this.image.Data[row, col, 0] = 0;
                }
            }
        }
        return this;
    }

    public ProcessedImage<TColor> Morphology(MorphOp operation, ElementShape shape, Size kernelSize) {
        var kernel = CvInvoke.GetStructuringElement(shape, kernelSize, new Point(-1, -1));
        this.image = this.image.MorphologyEx(operation, kernel, new Point(-1, -1), 1, BorderType.Default, default);
        return this;
    }

    public ProcessedImage<TColor> SaveImage(string filepath) {
        CvInvoke.Imwrite(filepath, this.image);
        return this;
    }

    public ProcessedImage<Gray> GenerateDistanceMask(DistType type, int maskSize) {
        Mat distanceMask = new();
        CvInvoke.DistanceTransform(
            this.image,          // IInputArray src,
            distanceMask,        // IOutputArray dst,
            this.distanceLabels, // IOutputArray labels,
            type,                // DistType distanceType,
            maskSize             // int maskSize,
            // DistLabelType labelType = DistLabelType.CComp
        );
        var iMask = distanceMask.ToImage<Gray, Byte>();
        return new ProcessedImage<Gray>(
            iMask,
            this.componentsLabels,
            this.componentsStats,
            this.componentsCentroids,
            this.labelsCount,
            this.distanceLabels,
            this.circles,
            this.lines
        );
    }

    public Image<TColor, Byte> GetImage() {
        return this.image;
    }

    public ProcessedImage<Gray> ConvertToGrayscale() {
        var gray = this.image.Convert<Gray, Byte>();
        return new ProcessedImage<Gray>(
            gray,
            this.componentsLabels,
            this.componentsStats,
            this.componentsCentroids,
            this.labelsCount,
            this.distanceLabels,
            this.circles,
            this.lines
        );
    }

    public ProcessedImage<Bgr> ConvertToColor() {
        var color = this.image.Convert<Bgr, Byte>();
        return new ProcessedImage<Bgr>(
            color,
            this.componentsLabels,
            this.componentsStats,
            this.componentsCentroids,
            this.labelsCount,
            this.distanceLabels,
            this.circles,
            this.lines
        );
    }

    public ProcessedImage<TColor> ThresholdBinary(TColor threshold, TColor max) {
        // dst(x,y) = max_value, if src(x,y) > threshold; 0, otherwise 
        this.image = this.image.ThresholdBinary(threshold, max);
        return this;
    }

    public ProcessedImage<TColor> ThresholdBinaryInv(TColor threshold, TColor max) {
        // dst(x,y) = 0, if src(x,y) > threshold; max_value, otherwise
        this.image = this.image.ThresholdBinaryInv(threshold, max);
        return this;
    }

    public ProcessedImage<TColor> FindCircles(TColor cannyThreshold, TColor accThreshold, double accResolution, double minDist, int minRadius = 0, int maxRadius = 0) {
        this.circles = this.image.HoughCircles(
            cannyThreshold, // TColor cannyThreshold,
            accThreshold,   // TColor accumulatorThreshold,
            accResolution,  // double dp,
            minDist,        // double minDist,
            minRadius,      // int minRadius = 0
            maxRadius       // int maxRadius = 0
        );
        return this;
    }

    public ProcessedImage<TColor> FindLines(double cannyThreshold, double cannyThresholdLinking, double rho, double theta, int threshold, double minLineWidth, double gapBetweenLines) {
        this.lines = this.image.HoughLines(
            cannyThreshold,        // double cannyThreshold,
            cannyThresholdLinking, // double cannyThresholdLinking,
            rho,                   // double rhoResolution,
            theta,                 // double thetaResolution,
            threshold,             // int threshold,
            minLineWidth,          // double minLineWidth,
            gapBetweenLines        // double gapBetweenLines
        );
        return this;
    }

    // fillPercent is 0.0-1.0
    public ProcessedImage<TColor> FindCirclesWithFillPercent(double fillPercent) {
        var result = new List<CircleF>();
        foreach(var circle in this.circles[0]) {
            Rectangle rect = Program.toRect(circle);
            var totalPixelCount = Math.PI * Math.Pow(circle.Radius, 2);
            int whitePixelCount = 0;
            for(int y = rect.Y; y < rect.Y + rect.Height; ++y) {
                for(int x = rect.X; x < rect.X + rect.Width; ++x) {
                    var p = new Point(x, y);
                    if(!Program.isPointInCircle(p, circle)) continue;
                    
                    var pixelColor = this.image.Data[y, x, 0];
                    if(pixelColor == 255) {
                        // white
                        whitePixelCount++;
                    }
                }
            }

            double whiteRatio = (double)whitePixelCount/(double)totalPixelCount;
            if(whiteRatio > fillPercent) {
                result.Add(circle);
            }
        }

        this.circles[0] = result.ToArray();
        return this;
    }

    public ProcessedImage<TColor> Difference(Image<TColor, Byte> img) {
        Mat diff = new();
        CvInvoke.AbsDiff(this.image, img, diff);
        var iDiff = diff.ToImage<TColor, Byte>();
        return new ProcessedImage<TColor>(
            iDiff,
            this.componentsLabels,
            this.componentsStats,
            this.componentsCentroids,
            this.labelsCount,
            this.distanceLabels,
            this.circles,
            this.lines
        );
    }

    public ProcessedImage<TColor> HighlightComponents() {
        var iCentroids = this.componentsCentroids.ToImage<Gray, Int32>();
        this.circles[0] = new CircleF[iCentroids.Rows];

        for(int i = 0; i < iCentroids.Rows; ++i) {
            int x = iCentroids.Data[i, 0, 0];
            int y = iCentroids.Data[i, 1, 0];
            int radius = 15;
            this.circles[0][i] = new CircleF(new PointF((float)x, (float)y), radius);
        }

        return this;
    }

    // if (min < pixel < max) pixel = white else pixel = black
    public ProcessedImage<TColor> FilterByDistance(int min, int max) {
        var result = this.image.Clone();
        for(int row = 0; row < this.image.Rows; ++row) {
            for(int col = 0; col < this.image.Cols; ++col) {
                int value = 0;
                // find distance value of a 3x3 area
                for(int x = row - 1; x <= row + 1; ++x) {
                    for(int y = col - 1; y <= col + 1; ++y) {
                        if(x < 0 || x >= this.image.Rows) continue;
                        if(y < 0 || y >= this.image.Cols) continue;
                        value += this.image.Data[x, y, 0];
                    }
                }
                
                if(min < value && value < max) {
                    // white
                    result.Data[row, col, 0] = 255;
                } else {
                    // black
                    result.Data[row, col, 0] = 0;
                }
            }
        }

        this.image = result;
        return this;
    }

    // if (pixel < min || max < pixel) pixel = white else pixel = black
    public ProcessedImage<TColor> FilterByDistanceInv(int min, int max) {
        var result = this.image.Clone();
        for(int row = 0; row < this.image.Rows; ++row) {
            for(int col = 0; col < this.image.Cols; ++col) {
                int value = 0;
                // find distance value of a 3x3 area
                for(int x = row - 1; x <= row + 1; ++x) {
                    for(int y = col - 1; y <= col + 1; ++y) {
                        if(x < 0 || x >= this.image.Rows) continue;
                        if(y < 0 || y >= this.image.Cols) continue;
                        value += this.image.Data[x, y, 0];
                    }
                }
                
                if(value < min || max < value) {
                    // white
                    result.Data[row, col, 0] = 255;
                } else {
                    // black
                    result.Data[row, col, 0] = 0;
                }
            }
        }

        this.image = result;
        return this;
    }

    public Image<Bgr, Byte> DrawCircles(Image<Bgr, Byte> img) {
        foreach (var circle in this.circles[0]) {
            img.Draw(circle, new Bgr(0, 0, 255), 3);
        }
        return img;
    }
}
// NOTE: the only holes not on the grayscale image are very big holes, make the max size bigger
//       all the other failed detection cases are present just not found by the circle detection, tweek parameters
