using System;
using System.Xml;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;

class ImageResult {
    public Image<Bgr, Byte> color;
    public Image<Bgr, Byte> blackAndWhite;
    public Image<Gray, int> stats;
}

class Program {
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

        // Draw circles around the contours
        for (int i = 0; i < contours.Size; i++) {
            // Get the bounding box for each contour
            Rectangle boundingBox = CvInvoke.BoundingRectangle(contours[i]);
            // Draw a circle around the bounding box center
            Point center = new Point(boundingBox.X + boundingBox.Width / 2, boundingBox.Y + boundingBox.Height / 2);
            int radius = Math.Max(boundingBox.Width, boundingBox.Height) / 2;
            CvInvoke.Circle(img2, center, radius, new MCvScalar(0, 255, 0), 2);
        }

        return img2;
    }
    
    static public string getInPathNoExt(string dir, int index) {
        string number = index < 10 ? "0" + index : "" + index;
        string fileNameNoExt = "01_missing_hole_" + number;
        // string fileName = fileNameNoExt + ".jpg";
        string path = dir + fileNameNoExt;
        return path;
    }

    static public string getOutPathNoExt(string dir, int index) {
        string outDirs = dir + index + "/";
        Directory.CreateDirectory(outDirs);
        
        string number = index < 10 ? "0" + index : "" + index;
        string fileNameNoExt = "01_missing_hole_" + number;
        // string outNameColor = fileNameNoExt + "_Color" + ".jpg";
        string outPath = outDirs + fileNameNoExt;
        return outPath;
    }

    static void Main(string[] args) {
        string outDir = "output/";
        string inDir = "PCB_DATASET/images/Missing_hole/";
        if (Directory.Exists(outDir)) Directory.Delete(outDir, true);
        Directory.CreateDirectory(outDir);
        
        for(int i = 1; i <= 20; ++i) {
            string inPath = getInPathNoExt(inDir, i) + ".jpg";

            var img = new Image<Bgr, Byte>(inPath);
            var res = DetectMissingHole(img);

            string outPathNoExt = getOutPathNoExt(outDir, i);
            string outPath = outPathNoExt + ".jpg";
            string outPathColor = outPathNoExt + "_Color" + ".jpg";
            
            CvInvoke.Imwrite(outPath, res.blackAndWhite);
            CvInvoke.Imwrite(outPathColor, res.color);
        }

        // string inPath = getInPathNoExt(inDir, 1) + ".jpg";
        // string outPathNoExt = getInPathNoExt(outDir, 1);
        // string outPath = outPathNoExt + ".jpg";
        // string outPathColor = outPathNoExt + "_Color" + ".jpg";

        // var img = new Image<Bgr, Byte>(inPath);
        // var res = DetectMissingHole(img);
        // CvInvoke.Imwrite(outPath, res.blackAndWhite);
        // CvInvoke.Imwrite(outPathColor, res.color);
    }

    static ImageResult DetectMissingHole(Image<Bgr, Byte> img, Int32 minArea = 300, Int32 maxArea = 900, Int32 threadholdVal = 40, Int32 threadholdInvVal = 128) {
        minArea = 200;
        maxArea = 600;
        var result = new ImageResult();
        result.color = img.Clone();
        result.blackAndWhite = img.Clone();

        var gray = img.Convert<Gray, byte>().SmoothGaussian(5);
        var thr = gray.ThresholdBinary(new Gray(threadholdVal), new Gray(255));
        var kernelCircle = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(8, 8), new Point(-1, -1)); // NOTE: maybe kernel size can be tuned
        var closeImage = thr.MorphologyEx(MorphOp.Close, kernelCircle, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
        var invImage = closeImage.ThresholdBinaryInv(new Gray(threadholdInvVal), new Gray(255));

        var LabelImage = new Mat();
        var stats = new Mat();
        var centroids = new Mat();
        CvInvoke.ConnectedComponentsWithStats(invImage, LabelImage, stats, centroids);

        Image<Gray, Int32> iStats = stats.ToImage<Gray, Int32>();
        Image<Gray, Byte> selectComp = new(LabelImage.Size);
        Image<Gray, Int32> LabelImageIM = LabelImage.ToImage<Gray, Int32>();
        result.stats = LabelImageIM.Clone();

        for (int row = 0; row < LabelImage.Rows; row++) {
            for (int col = 0; col < LabelImage.Cols; col++) {
                Int32 componentIdx = LabelImageIM.Data[row, col, 0];
                if (componentIdx == 0) continue;
                Int32 componentArea = iStats.Data[componentIdx, 4, 0];

                if (minArea < componentArea && componentArea < maxArea) {
                    selectComp.Data[row, col, 0] = 255;
                } else {
                    selectComp.Data[row, col, 0] = 0;
                }
            }
        }

        // Mat kernelCircle2 = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(14, 14), new Point(-1, -1));
        // CvInvoke.MorphologyEx(selectComp, selectComp, MorphOp.Open, kernelCircle2, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
        // CvInvoke.Imwrite("_06_open_image.png", selectComp);

        // Image<Bgr, byte> outputImage = img.Clone();
        // for (int row = 0; row < LabelImage.Rows; row++) {
        //     for (int col = 0; col < LabelImage.Cols; col++) {
        //         if (selectComp.Data[row, col, 0] > 128) outputImage.Data[row, col, 2] = 255;
        //     }
        // }

        selectComp = selectComp.SmoothGaussian(5);
        // 255, 50, 5, 1, 0, 50 - found all 3
        var circles = selectComp.HoughCircles(
            new Gray(255),  // Canny küszöb
            new Gray(50),  // A kör középpontjának küszöbértéke
            5,            // Akkumulátor felbontása
            1,            // Minimum távolság a körök között
            0,
            20
        );
        
        // NOTE: iterate over all the circles, iterate the pixels inside the circle,
        //       get the percent of the circle filled with white pixels, if below  a threashold discard the circle
        int index = 0;
        var filteredCircles = new List<CircleF>();
        foreach(var circle in circles[0]) {
            Rectangle rect = toRect(circle);
            // int totalPixelCount = rect.Width * rect.Height;
            var totalPixelCount = Math.PI * Math.Pow(circle.Radius, 2);
            double whitePixelCount = 0;
            for(int y = rect.Y; y < rect.Y + rect.Height; ++y) {
                for(int x = rect.X; x < rect.X + rect.Width; ++x) {
                    var p = new Point(x, y);
                    if(!isPointInCircle(p, circle)) continue;
                    
                    var pixelColor = selectComp.Data[y, x, 0];
                    if(pixelColor == 255) {
                        // white
                        whitePixelCount++;
                    } else if(pixelColor == 0) {
                        // black
                    } else {
                        // Console.WriteLine($"Pixel color not black or white: {pixelColor}");
                    }
                }
            }

            var whiteRatio = whitePixelCount/totalPixelCount;
            if(whiteRatio > 0.4) {
                filteredCircles.Add(circle);
                // Console.WriteLine($"added circle {index} with ratio: {whiteRatio}");
            }

            // Console.WriteLine($"total pixel count: {totalPixelCount}");
            // Console.WriteLine($"white pixel count: {whitePixelCount}");
            // Console.WriteLine($"percent of circle {index} is {whiteRatio}");
            index++;
        }

        result.blackAndWhite = selectComp.Convert<Bgr, byte>();
        result.color = img.Convert<Bgr, byte>();
        foreach (var circle in filteredCircles) {
            result.blackAndWhite.Draw(circle, new Bgr(0, 0, 255), 3);
            result.color.Draw(circle, new Bgr(0, 0, 255), 3);
        }

        return result;
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
}

// NOTE: the only holes not on the grayscale image are very big holes, make the max size bigger
//       all the other failed detection cases are present just not found by the circle detection, tweek parameters
