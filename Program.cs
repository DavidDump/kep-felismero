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
    
    static void Main(string[] args) {
        string outDir = "output/";
        if (Directory.Exists(outDir)) Directory.Delete(outDir, true);
        Directory.CreateDirectory(outDir);
        
        for(int i = 1; i <= 20; ++i) {
            string dir = "PCB_DATASET/images/Missing_hole/";
            string number = i < 10 ? "0" + i : "" + i;
            string fileNameNoExt = "01_missing_hole_" + number;
            string fileName = fileNameNoExt + ".jpg";
            string path = dir + fileName;

            var img = new Image<Bgr, Byte>(path);
            var res = DetectMissingHole(img);

            string outDirs = outDir + i + "/";
            Directory.CreateDirectory(outDirs);
            string outNameColor = fileNameNoExt + "_Color" + ".jpg";
            string outPath = outDirs + fileName;
            string outPathColor = outDirs + outNameColor;
            CvInvoke.Imwrite(outPath, res.blackAndWhite);
            CvInvoke.Imwrite(outPathColor, res.color);
        }
    }

    static ImageResult DetectMissingHole(Image<Bgr, Byte> img, Int32 minArea = 300, Int32 maxArea = 900, Int32 threadholdVal = 40, Int32 threadholdInvVal = 128) {
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
        
        result.blackAndWhite = selectComp.Convert<Bgr, byte>();
        result.color = img.Convert<Bgr, byte>();
        foreach (var circle in circles[0]) {
            result.blackAndWhite.Draw(circle, new Bgr(0, 0, 255), 3);
            result.color.Draw(circle, new Bgr(0, 0, 255), 3);
        }

        return result;
    }
}
