using System;
using System.Xml;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;

public class Setting {
    public Int32 minArea;
    public Int32 maxArea;
    public Int32 threadholdVal;
    public Int32 threadholdInvVal;

    public Setting(Int32 minArea = 300, Int32 maxArea = 900, Int32 threadholdVal = 40, Int32 threadholdInvVal = 128) {
        this.minArea = minArea;
        this.maxArea = maxArea;
        this.threadholdVal = threadholdVal;
        this.threadholdInvVal = threadholdInvVal;
    }

    public string toString() {
        return $"({this.minArea}, {this.maxArea}, {this.threadholdVal}, {this.threadholdInvVal})";
    }
}

class Program {
    static private Int32 minArea_LOW = 275;
    static private Int32 minArea_HIGH = 325;
    static private Int32 minArea_COUNT = (minArea_HIGH - minArea_LOW);
    static private Int32 maxArea_LOW = 875;
    static private Int32 maxArea_HIGH = 925;
    static private Int32 maxArea_COUNT = (maxArea_HIGH - maxArea_LOW);
    static private Int32 threadholdVal_LOW = 30;
    static private Int32 threadholdVal_HIGH = 50;
    static private Int32 threadholdVal_COUNT = (threadholdVal_HIGH - threadholdVal_LOW);
    static private Int32 threadholdInvVal_LOW = 110;
    static private Int32 threadholdInvVal_HIGH = 140;
    static private Int32 threadholdInvVal_COUNT = (threadholdInvVal_HIGH - threadholdInvVal_LOW);

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
    
    static void Main(string[] args) {
        string outDir = "output/";
        new DirectoryInfo(outDir).Delete(true);
        Directory.CreateDirectory(outDir);

        int totalImgCount = minArea_COUNT * maxArea_COUNT * threadholdVal_COUNT * threadholdInvVal_COUNT * 20;
        int imageProgress = 0;
        var stats = new List<Dictionary<Setting, int>>();
        stats.Add(new Dictionary<Setting, int>());
        for(int i = 1; i <= 20; ++i) {
            string dir = "PCB_DATASET/images/Missing_hole/";
            string number = i < 10 ? "0" + i : "" + i;
            string fileNameNoExt = "01_missing_hole_" + number;
            string path = dir + fileNameNoExt + ".jpg";

            string xmlDir = "PCB_DATASET/Annotations/Missing_hole/";
            string xmlPath = xmlDir + fileNameNoExt + ".xml";
            byte[] xmlBytes = File.ReadAllBytes(xmlPath);
            string xmlStr = System.Text.Encoding.Default.GetString(xmlBytes);
            XmlDocument doc = new XmlDocument();
            doc.LoadXml(xmlStr);
            XmlElement root = doc.DocumentElement;
            List<Rectangle> rects = new();
            foreach (XmlElement element in root.GetElementsByTagName("object")) {
                try {
                    var box = element["bndbox"];
                    var xmin = Int32.Parse(box["xmin"].InnerText);
                    var ymin = Int32.Parse(box["ymin"].InnerText);
                    var xmax = Int32.Parse(box["xmax"].InnerText);
                    var ymax = Int32.Parse(box["ymax"].InnerText);
                    Rectangle boundingBox = new Rectangle(xmin, ymin, xmax - xmin, ymax - ymin);
                    rects.Add(boundingBox);
                } catch (Exception e) {
                    Console.WriteLine(e);
                }
            }

            Setting bestSet = new Setting();
            int bestVal = 0;
            stats.Add(new Dictionary<Setting, int>());
            for(Int32 minArea = minArea_LOW; minArea <= minArea_HIGH; ++minArea) {
                for(Int32 maxArea = maxArea_LOW; maxArea <= maxArea_HIGH; ++maxArea) {
                    for(Int32 threadholdVal = threadholdVal_LOW; threadholdVal <= threadholdVal_LOW; ++threadholdVal) {
                        for(Int32 threadholdInvVal = threadholdInvVal_LOW; threadholdInvVal <= threadholdInvVal_HIGH; ++threadholdInvVal) {
                            var setting = new Setting(minArea, maxArea, threadholdVal, threadholdInvVal);
                            var foundBBoxs = DetectMissingHole(path, i, outDir, setting);

                            int found = 0;
                            foreach(var r1 in rects) {
                                foreach(var r2 in foundBBoxs) {
                                    if(!Rectangle.Intersect(r1, r2).IsEmpty) {
                                        found++;
                                    }
                                }
                            }
                            stats[i].Add(setting, found);
                            if(found > bestVal) bestSet = setting;

                            Console.Write($"\r[{imageProgress}/{totalImgCount}]");
                            imageProgress++;
                        }
                    }
                }
            }

            Console.WriteLine($"Best setting for image index {i}: {bestSet.toString()}");
        }
    }

    static List<Rectangle> DetectMissingHole(string filepath, int count, string outDir, Setting setting) {
        List<Rectangle> result = new();
        var img = new Image<Bgr, Byte>(filepath);

        var gray = img.Convert<Gray, byte>().SmoothGaussian(5);
        var thr = gray.ThresholdBinary(new Gray(setting.threadholdVal), new Gray(255));
        var kernelCircle = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(8, 8), new Point(-1, -1)); // NOTE: maybe kernel size can be tuned
        var closeImage = thr.MorphologyEx(MorphOp.Close, kernelCircle, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
        var invImage = closeImage.ThresholdBinaryInv(new Gray(setting.threadholdInvVal), new Gray(255));

        var LabelImage = new Mat();
        var stats = new Mat();
        var centroids = new Mat();
        CvInvoke.ConnectedComponentsWithStats(invImage, LabelImage, stats, centroids);

        Image<Gray, Int32> iStats = stats.ToImage<Gray, Int32>();
        Image<Gray, Byte> selectComp = new(LabelImage.Size);
        Image<Gray, Int32> LabelImageIM = LabelImage.ToImage<Gray, Int32>();

        for (int row = 0; row < LabelImage.Rows; row++) {
            for (int col = 0; col < LabelImage.Cols; col++) {
                Int32 componentIdx = LabelImageIM.Data[row, col, 0];
                if (componentIdx == 0) continue;
                Int32 componentArea = iStats.Data[componentIdx, 4, 0];

                if (setting.minArea < componentArea && componentArea < setting.maxArea) {
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

        var circles = selectComp.HoughCircles(
            new Gray(500),  // Canny küszöb
            new Gray(60),  // A kör középpontjának küszöbértéke
            1.5,            // Akkumulátor felbontása
            1            // Minimum távolság a körök között
        );
        var outputImage = img;
        foreach (var circle in circles[0]) {
            outputImage.Draw(circle, new Bgr(0, 0, 255), 3);
            var center = circle.Center;
            var radius = circle.Radius;
            Rectangle boundingBox = new Rectangle((int)(center.X - radius), (int)(center.Y - radius), (int)(radius * 2), (int)(radius * 2));
            result.Add(boundingBox);
        }

        #if SOBEL
        var sobel = closeImage.Sobel(1, 1, 3);
        CvInvoke.Imwrite("_03_sobel_image.png", sobel);
        #endif

        #if ASDF
        CircleF[] circles = CvInvoke.HoughCircles(
            gray,                 // Szürkeárnyalatos képet használunk
            HoughModes.Gradient,  // Hough Circle módszer
            4,                    // Akkumulátor felbontása
            50.0,                 // Minimum távolság a körök között
            500.0,                // Canny küszöb
            30.0,                 // A kör középpontjának küszöbértéke
            10,                   // Minimum sugár
            30                    // Maximum sugár
        );

        foreach (var circle in circles) {
            Point center = new Point((int)circle.Center.X, (int)circle.Center.Y);
            int radius = (int)circle.Radius;
            CvInvoke.Circle(outputImage, center, radius, new MCvScalar(0, 0, 255), 5);
            continue;

            Rectangle boundingBox = new Rectangle(center.X - radius, center.Y - radius, radius * 2, radius * 2);
            Mat circleROI = new Mat(closeImage, boundingBox);
            Image<Gray, byte> circleArea = circleROI.ToImage<Gray, byte>();

            int totalPixelCount = boundingBox.Width * boundingBox.Height;
            int blackPixelCount = totalPixelCount - CvInvoke.CountNonZero(circleArea);

            double blackPixelRatio = (double)blackPixelCount / totalPixelCount;
            if (center.X >= 0 && center.X < closeImage.Width && center.Y >= 0 && center.Y < closeImage.Height) {
                byte centerIntensity = circleArea.Data[center.Y - boundingBox.Y, center.X - boundingBox.X, 0];

                if (centerIntensity == 255 && blackPixelRatio > 0.7) {
                    CvInvoke.Circle(outputImage, center, radius, new MCvScalar(0, 0, 255), 5);
                }
            }
        }
        #endif

        string outDirs = outDir + count + "/";
        Directory.CreateDirectory(outDirs);
        string outName = setting.minArea + "_" + setting.maxArea + "_" + setting.threadholdVal + "_" + setting.threadholdInvVal + ".jpg";
        string outPath = outDirs + outName;
        // CvInvoke.Imwrite(outPath, outImage);
        return result;
    }
}
