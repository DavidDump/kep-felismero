using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;

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
        Mat masterImg = CvInvoke.Imread("PCB_DATASET/PCB_USED/01.JPG", ImreadModes.Color);
        
        for(int i = 1; i <= 20; ++i) {
            string dir = "PCB_DATASET/images/Missing_hole/";
            string number = i < 10 ? "0" + i : "" + i;
            string name = "01_missing_hole_" + number + ".jpg";
            string path = dir + name;
            Mat img = CvInvoke.Imread(path, ImreadModes.Color);
            Mat outImg = CompareImages(masterImg, img);

            string outDir = "output/";
            string outName = "" + i + ".jpg";
            string outPath = outDir + outName;
            CvInvoke.Imwrite(outPath, outImg);
        }
    }
}
