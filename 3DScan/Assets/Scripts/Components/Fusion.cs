using UnityEngine;
using OpenCVForUnity.CoreModule;
using System.Collections.Generic;
using OpenCVForUnity.Features2dModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using System.Linq;
using System;
using UnityEngine.UI;
using System.IO;
using System.Text;
using Unity.VisualScripting;
using TMPro;
using JetBrains.Annotations;

public class Fusion : MonoBehaviour
{
    private Dictionary<Vector3Int, (float tsdf, float weight, Color color)> tsdfGrid = new();
    public float voxelSize = 0.01f;
    public float truncation = 0.03f;
    float quantityQualityNorm = 30f;
    float distanceQualityDecay = 2.5f;
    PointCloudRenderer PCDRenderer;
    List<Vector3> PointCloud = new List<Vector3>();
    List<Color> PointCloudColor = new List<Color>();
    List<Vector3> RenderPointCloud = new List<Vector3>();
    List<Color> RenderPointCloudColor = new List<Color>();
    bool PointCloudMode = false; //false  = TSDF, true = Point Cloud
    bool StopUpdatingTSDF = false;
    Texture2D prevRGBFrame = null;
    Mat prevRGBMat = new Mat();
    Texture2D prevDepthFrame = null;
    Vector3[] prevDepthPoints = null;
    private Pose prevDepthCameraPose;
    private ORB detector;
    private DescriptorMatcher matcher;
    private MatOfKeyPoint prevKeypoints = new MatOfKeyPoint();
    private Mat prevDescriptors = new Mat();
    private List<Vector3> fusedPointCloud = new List<Vector3>();
    private List<Color> fusedColors = new List<Color>();
    public RawImage TextureVisualizer;
    public TextMeshProUGUI ModeButtonText;
    public TextMeshProUGUI StopTSDFButtonText;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        PCDRenderer = FindFirstObjectByType<PointCloudRenderer>();
        detector = ORB.create();
        detector.setMaxFeatures(1000);
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
    }

    // Update is called once per frame
    void Update()
    {

    }
    public void SwtichMode()
    {
        PointCloudMode = !PointCloudMode;
        RenderPCD();
    }
    public void SwtichTSDFMode()
    {
        StopUpdatingTSDF = !StopUpdatingTSDF;
        StopTSDFButtonText.text = StopUpdatingTSDF ? "TSDF Disabled" : "TSDF Enabled";
        //RenderPCD();
    }
    void RenderPCD()
    {
        if (PointCloudMode)
        {
            RenderPointCloud = PointCloud;
            RenderPointCloudColor = PointCloudColor;
            ModeButtonText.text = "Point Cloud";
        }
        else
        {
            ExtractSurfacePointsFromTSDF(0, voxelSize * 0.5f, 0.5f);
            ModeButtonText.text = "TSDF";
        }
        PCDRenderer.Render(RenderPointCloud.ToArray(), RenderPointCloudColor.ToArray());
    }
    public void Clear()
    {
        PointCloud.Clear();
        PointCloudColor.Clear();
        RenderPointCloud.Clear();
        RenderPointCloudColor.Clear();
        tsdfGrid.Clear();
        prevRGBFrame = null;
        prevDepthFrame = null;
    }


    public void VisualizeByTexture(Texture2D texture)
    {
        if (texture == null || TextureVisualizer == null) return;
        TextureVisualizer.texture = texture;
        TextureVisualizer.GetComponent<RectTransform>().sizeDelta = new Vector2(texture.width, texture.height);
    }

    public void PCDFusion(Vector3[] depthPoints, Color[] colors, Texture2D depthImage, Texture2D rgbImage, Pose depthCameraPose, Pose rgbCameraPose)
    {
        Mat currentGrayMat = new Mat(rgbImage.height, rgbImage.width, CvType.CV_8UC1);
        Mat currentRGBMat = new Mat(rgbImage.height, rgbImage.width, CvType.CV_8UC3);
        Utils.texture2DToMat(rgbImage, currentRGBMat);
        Imgproc.cvtColor(currentRGBMat, currentGrayMat, Imgproc.COLOR_RGB2GRAY);

        Texture2D grayTexture = new Texture2D(currentGrayMat.cols(), currentGrayMat.rows(), TextureFormat.R8, false);
        Utils.matToTexture2D(currentGrayMat, grayTexture);
        VisualizeByTexture(grayTexture);

        MatOfKeyPoint currentKeypoints = new MatOfKeyPoint();
        Mat currentDescriptors = new Mat();

        if (prevRGBFrame != null && prevDescriptors.empty() == false)
        {
            // SaveDepthImage(depthPoints,rgbImage.width,rgbImage.height,$"curr_desc{DateTime.Now:dd-MM-yyyy-HH-mm-ss}.png");
            // SaveDepthImage(prevDepthPoints,rgbImage.width,rgbImage.height,$"prev_desc{DateTime.Now:dd-MM-yyyy-HH-mm-ss}.png"); 
            detector.detectAndCompute(currentGrayMat, new Mat(), currentKeypoints, currentDescriptors);
            Debug.Log($"detected keypoints,{currentKeypoints.rows()},{currentDescriptors.empty()}");
            if (!currentDescriptors.empty())
            {
                MatOfDMatch matches = new MatOfDMatch();
                matcher.match(prevDescriptors, currentDescriptors, matches);
                List<DMatch> matchList = matches.toList();
                Debug.Log($"Found {matchList.Count} matches");
                double maxDist = 0;
                double minDist = 100;
                foreach (DMatch match in matchList)
                {
                    double dist = match.distance;
                    if (dist < minDist) minDist = dist;
                    if (dist > maxDist) maxDist = dist;
                }
                List<DMatch> goodMatches = matchList.FindAll(m => m.distance <= Mathf.Max(2 * (float)minDist, 0.02f));
                Debug.Log($"Good matches: {goodMatches.Count}");
                List<Vector3> sourcePoints = new List<Vector3>();
                List<Vector3> targetPoints = new List<Vector3>();

                KeyPoint[] prevKPArray = prevKeypoints.toArray();
                KeyPoint[] currentKPArray = currentKeypoints.toArray();
                VisualizeAndSaveMatches(
                    prevRGBMat, prevKeypoints,
                    currentRGBMat, currentKeypoints,
                    goodMatches,
                    $"feature_matches{DateTime.Now:dd-MM-yyyy-HH-mm-ss}.png"
                );
                VisualizeAndSaveDepthMatches(prevDepthPoints,
                depthPoints,
                rgbImage.width,
                rgbImage.height,
                goodMatches,
                prevKeypoints,
                currentKeypoints, $"prev_depth_matches{DateTime.Now:dd-MM-yyyy-HH-mm-ss}.png",
                $"curr_depth_matches{DateTime.Now:dd-MM-yyyy-HH-mm-ss}.png");
                foreach (DMatch match in goodMatches)
                {
                    Point prevPoint = prevKPArray[match.queryIdx].pt;
                    prevPoint.y = rgbImage.height - prevPoint.y;
                    // Debug.Log("PrevP:" + prevPoint.x + "," + prevPoint.y);
                    Vector3 prevWorldPoint = GetWorldPointFromDepth(prevPoint, prevDepthPoints, rgbImage.width, prevDepthCameraPose);
                    if (prevWorldPoint == Vector3.zero) continue;

                    Point currentPoint = currentKPArray[match.trainIdx].pt;
                    currentPoint.y = rgbImage.height - currentPoint.y;
                    // Debug.Log("CurrP" + currentPoint.x + "," + currentPoint.y);
                    Vector3 currentLocalPoint = GetLocalPointFromDepth(currentPoint, depthPoints, rgbImage.width);
                    if (currentLocalPoint == Vector3.zero) continue;

                    sourcePoints.Add(prevWorldPoint);
                    targetPoints.Add(currentLocalPoint);
                }

                Debug.Log("source&target: " + sourcePoints.Count + "|" + targetPoints.Count);
                if (sourcePoints.Count >= 4)
                {
                    Vector3 sourceCentroid = CalculateCentroid(sourcePoints);
                    Vector3 targetCentroid = CalculateCentroid(targetPoints);
                    Matrix4x4 H = Matrix4x4.zero;
                    for (int i = 0; i < sourcePoints.Count; i++)
                    {
                        Vector3 s = sourcePoints[i] - sourceCentroid;
                        Vector3 t = targetPoints[i] - targetCentroid;
                        H.m00 += s.x * t.x; H.m01 += s.x * t.y; H.m02 += s.x * t.z;
                        H.m10 += s.y * t.x; H.m11 += s.y * t.y; H.m12 += s.y * t.z;
                        H.m20 += s.z * t.x; H.m21 += s.z * t.y; H.m22 += s.z * t.z;
                    }


                    Matrix4x4 R_Matrix = Matrix4x4.identity;
                    Quaternion optimalRotation = Quaternion.identity;

                    var (U, S, Vt) = ComputeSVD3x3(H);
                    Matrix4x4 V = Vt.transpose;
                    R_Matrix = V * U.transpose;

                    if (R_Matrix.determinant < 0)
                    {

                        Matrix4x4 M = Matrix4x4.identity;
                        M.m22 = -1;
                        R_Matrix = V * M * U.transpose;
                    }
                    optimalRotation = QuaternionFromMatrix(R_Matrix);

                    Vector3 optimalTranslation = targetCentroid - R_Matrix.MultiplyPoint(sourceCentroid);

                    Pose deltaPose = new Pose(optimalTranslation, optimalRotation);

                    Quaternion refinedRotation = deltaPose.rotation * prevDepthCameraPose.rotation;
                    Vector3 refinedPosition = deltaPose.rotation * prevDepthCameraPose.position + deltaPose.position;

                    float matchValue = CalculateMatchQuality(goodMatches, sourcePoints.Count);
                    Pose refinedDepthPose = new Pose(
                        Vector3.Lerp(depthCameraPose.position, refinedPosition, matchValue),
                        Quaternion.Slerp(depthCameraPose.rotation, refinedRotation, matchValue));
                    Vector3[] refinedWorldPoints = new Vector3[depthPoints.Length];
                    for (int i = 0; i < depthPoints.Length; i++)
                    {
                        Vector3 worldPoint = depthPoints[i] == Vector3.zero ? Vector3.zero : refinedDepthPose.rotation * depthPoints[i] + refinedDepthPose.position;
                        refinedWorldPoints[i] = worldPoint;
                    }

                    FuseWithVoxelGrid(refinedWorldPoints, colors, refinedDepthPose);
                    prevDepthCameraPose = refinedDepthPose;
                }
                else
                {
                    Debug.LogWarning("Not enough correspondences for ICP. Using original pose.");
                    TransformAndFuse(depthPoints, colors, depthCameraPose);
                    prevDepthCameraPose = depthCameraPose;
                }
            }
        }
        else
        {
            Debug.Log("First frame, initializing.");
            TransformAndFuse(depthPoints, colors, depthCameraPose);
            detector.detectAndCompute(currentRGBMat, new Mat(), currentKeypoints, currentDescriptors);
            Debug.Log($"detected keypoints,{currentKeypoints.rows()},{currentDescriptors.empty()}");
            prevDepthCameraPose = depthCameraPose;
        }
        prevRGBFrame = rgbImage;
        prevDepthFrame = depthImage;
        prevRGBMat = currentRGBMat;
        currentKeypoints.copyTo(prevKeypoints);
        currentDescriptors.copyTo(prevDescriptors);
        prevDepthPoints = (Vector3[])depthPoints.Clone();
        RenderPCD();
    }
    public static Texture2D ConvertToGrayscale(Texture2D rgbTexture)
    {
        int width = rgbTexture.width;
        int height = rgbTexture.height;
        Texture2D grayTexture = new Texture2D(width, height, TextureFormat.R8, false);

        Color[] pixels = rgbTexture.GetPixels();
        Color[] grayPixels = new Color[pixels.Length];

        for (int i = 0; i < pixels.Length; i++)
        {
            float gray = 0.299f * pixels[i].r + 0.587f * pixels[i].g + 0.114f * pixels[i].b;
            grayPixels[i] = new Color(gray, gray, gray, 1.0f);
        }
        // Debug.Log(grayPixels[123]);
        grayTexture.SetPixels(grayPixels);
        grayTexture.Apply();
        return grayTexture;
    }
    private float CalculateMatchQuality(List<DMatch> goodMatches, int srcCount)
    {
        float quantityQuality = Mathf.Clamp01(srcCount / quantityQualityNorm);
        float avgDistance = goodMatches.Average(m => m.distance);
        float distanceQuality = Mathf.Exp(-distanceQualityDecay * avgDistance);
        return quantityQuality * distanceQuality;
    }
    private Vector3 GetWorldPointFromDepth(Point pixel, Vector3[] depthData, int width, Pose cameraPose)
    {
        // Debug.Log("depth length: " + depthData.Length);
        // Debug.Log("now pixel:" + pixel.x + "," + pixel.y + ", index:" + ((int)pixel.y * width + (int)pixel.x));
        Vector3 depthValue = depthData[(int)pixel.y * width + (int)pixel.x];
        // Debug.Log("prev World DepthValue:" + depthValue);
        // Debug.Log("neighbors: " + depthData[Math.Clamp((int)pixel.y * width + (int)pixel.x - 1, 0, depthData.Length - 1)] + "|" + depthData[Math.Clamp((int)pixel.y * width + (int)pixel.x + 1, 0, depthData.Length - 1)]);
        return Vector3.Distance(depthValue, Vector3.zero) < 0.01f ? Vector3.zero : TransformPointToWorld(depthValue, cameraPose);
    }
    private Vector3 GetLocalPointFromDepth(Point pixel, Vector3[] depthData, int width)
    {
        Vector3 depthValue = depthData[(int)pixel.y * width + (int)pixel.x];
        // Debug.Log("now local DepthValue:" + depthValue);
        // Debug.Log("neighbors: " + depthData[Math.Clamp((int)pixel.y * width + (int)pixel.x - 1, 0, depthData.Length - 1)] + "|" + depthData[Math.Clamp((int)pixel.y * width + (int)pixel.x + 1, 0, depthData.Length - 1)]);
        return depthValue;
    }

    private Vector3 CalculateCentroid(List<Vector3> points)
    {
        Vector3 centroid = Vector3.zero;
        foreach (Vector3 p in points) centroid += p;
        return centroid / points.Count;
    }

    private void TransformAndFuse(Vector3[] localPoints, Color[] colors, Pose pose)
    {
        Vector3[] worldPoints = TransformPointCloud(localPoints, pose);
        FuseWithVoxelGrid(worldPoints, colors, pose);
    }
    private void FuseWithPointCloud(Vector3[] newPoints, Color[] newColors)
    {
        // Debug.Log($"Fusing {newPoints.Length} new points with {newColors.Length} colors.");
        for (int i = 0; i < newPoints.Length; i++)
        {
            if (newPoints[i] != Vector3.zero)
            {
                PointCloud.Add(newPoints[i]);
                PointCloudColor.Add(newColors[i]);
            }
        }
    }
    private void FuseWithVoxelGrid(Vector3[] newPoints, Color[] newColors, Pose cameraPose)
    {
        FuseWithPointCloud(newPoints, newColors);
        if (StopUpdatingTSDF) return;
        // Debug.Log($"Fusing {newPoints.Length} new points with {newColors.Length} colors (TSDF ray update).");
        Vector3 cameraOrigin = cameraPose.position;

        for (int i = 0; i < newPoints.Length; i++)
        {
            if (newPoints[i] == Vector3.zero) continue;
            Vector3 surfPt = newPoints[i];

            Vector3 dir = (surfPt - cameraOrigin).normalized;
            float surfaceDist = (surfPt - cameraOrigin).magnitude;

            float maxDist = surfaceDist + truncation;
            float step = voxelSize * 0.25f;
            for (float t = Mathf.Max(0, surfaceDist - truncation); t <= maxDist; t += step)
            {
                Vector3 pos = cameraOrigin + dir * t;
                Vector3Int voxel = new Vector3Int(
                    Mathf.FloorToInt(pos.x / voxelSize),
                    Mathf.FloorToInt(pos.y / voxelSize),
                    Mathf.FloorToInt(pos.z / voxelSize)
                );
                Vector3 voxelCenter = new Vector3(
                    (voxel.x + 0.5f) * voxelSize,
                    (voxel.y + 0.5f) * voxelSize,
                    (voxel.z + 0.5f) * voxelSize
                );
                float sdf = surfaceDist - (voxelCenter - cameraOrigin).magnitude;
                sdf = Mathf.Clamp(sdf, -truncation, truncation);

                if (Mathf.Abs(sdf) > truncation) continue;
                if (tsdfGrid.TryGetValue(voxel, out var entry))
                {
                    float wOld = entry.weight;
                    float wNew = wOld + 1f;
                    float tsdf = (entry.tsdf * wOld + sdf) / wNew;
                    Color color = (entry.color * (wOld + 1) + newColors[i]) / (1 + wNew);
                    tsdfGrid[voxel] = (tsdf, wNew, color);
                }
                else
                {
                    tsdfGrid[voxel] = (sdf, 0.5f, newColors[i]);
                }
            }
        }
    }
    public void ExtractSurfacePointsFromTSDF(float isoLevel = 0.0f, float tsdfThreshold = 0.005f, float minWeight = 0.5f)
    {
        List<Vector3> points = new List<Vector3>();
        List<Color> colors = new List<Color>();

        foreach (var kvp in tsdfGrid)
        {
            var voxel = kvp.Key;
            var (tsdf, weight, color) = kvp.Value;
            if (weight >= minWeight && Mathf.Abs(tsdf - isoLevel) < tsdfThreshold)
            {
                Vector3 point = new Vector3(
                    (voxel.x + 0.5f) * voxelSize,
                    (voxel.y + 0.5f) * voxelSize,
                    (voxel.z + 0.5f) * voxelSize
                );
                points.Add(point);
                colors.Add(color);
            }
        }
        RenderPointCloud = points;
        RenderPointCloudColor = colors;
    }
    Vector3 TransformPointToWorld(Vector3 point, Matrix4x4 transformation)
    {
        return transformation.MultiplyPoint(point);
    }
    Vector3 TransformPointToWorld(Vector3 point, Pose pose)
    {
        Matrix4x4 transformation = Matrix4x4.TRS(pose.position, pose.rotation, Vector3.one);
        return transformation.MultiplyPoint(point);
    }
    Vector3[] TransformPointCloud(Vector3[] points, Matrix4x4 transformation)
    {
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = points[i] == Vector3.zero ? Vector3.zero : TransformPointToWorld(points[i], transformation);
        }
        return points;
    }
    Vector3[] TransformPointCloud(Vector3[] points, Pose pose)
    {
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = points[i] == Vector3.zero ? Vector3.zero : TransformPointToWorld(points[i], pose);
        }
        return points;
    }
    private (Matrix4x4 U, Matrix4x4 S, Matrix4x4 Vt) ComputeSVD3x3(Matrix4x4 H)
    {
        float[,] hMatrix = new float[3, 3] {
        { H.m00, H.m01, H.m02 },
        { H.m10, H.m11, H.m12 },
        { H.m20, H.m21, H.m22 }
    };

        Matrix<float> matrix = DenseMatrix.OfArray(hMatrix);

        var svd = matrix.Svd(true);

        Matrix4x4 U = MatrixFromMathNet(svd.U);
        Matrix4x4 S = SingularValueMatrixFromVector(svd.S);
        Matrix4x4 Vt = MatrixFromMathNet(svd.VT);

        return (U, S, Vt);
    }
    private Quaternion QuaternionFromMatrix(Matrix4x4 m)
    {
        Quaternion q = new Quaternion();

        q.w = Mathf.Sqrt(Mathf.Max(0, 1 + m.m00 + m.m11 + m.m22)) / 2;
        q.x = Mathf.Sqrt(Mathf.Max(0, 1 + m.m00 - m.m11 - m.m22)) / 2;
        q.y = Mathf.Sqrt(Mathf.Max(0, 1 - m.m00 + m.m11 - m.m22)) / 2;
        q.z = Mathf.Sqrt(Mathf.Max(0, 1 - m.m00 - m.m11 + m.m22)) / 2;

        q.x *= Mathf.Sign(q.x * (m.m21 - m.m12));
        q.y *= Mathf.Sign(q.y * (m.m02 - m.m20));
        q.z *= Mathf.Sign(q.z * (m.m10 - m.m01));

        q.Normalize();
        return q;
    }

    private Matrix4x4 MatrixFromMathNet(Matrix<float> mathNetMatrix)
    {
        if (mathNetMatrix.RowCount == 3 && mathNetMatrix.ColumnCount == 3)
        {
            return new Matrix4x4(
                new Vector4((float)mathNetMatrix[0, 0], (float)mathNetMatrix[0, 1], (float)mathNetMatrix[0, 2], 0),
                new Vector4((float)mathNetMatrix[1, 0], (float)mathNetMatrix[1, 1], (float)mathNetMatrix[1, 2], 0),
                new Vector4((float)mathNetMatrix[2, 0], (float)mathNetMatrix[2, 1], (float)mathNetMatrix[2, 2], 0),
                new Vector4(0, 0, 0, 1)
            );
        }
        return Matrix4x4.identity;
    }
    private Matrix4x4 SingularValueMatrixFromVector(Vector<float> singularValues)
    {
        Matrix4x4 S = Matrix4x4.identity;
        for (int i = 0; i < Mathf.Min(3, singularValues.Count); i++)
        {
            S[i, i] = (float)singularValues[i];
        }
        return S;
    }
    public static void WriteMatToCSV(Mat mat, string filePath)
    {
        using (System.IO.StreamWriter writer = new System.IO.StreamWriter(filePath))
        {
            int rows = mat.rows();
            int cols = mat.cols();
            int channels = mat.channels();

            for (int i = 0; i < rows; i++)
            {
                List<string> rowValues = new List<string>();
                for (int j = 0; j < cols; j++)
                {
                    if (channels == 1)
                    {
                        double val = mat.get(i, j)[0];
                        rowValues.Add(val.ToString("G3"));
                    }
                    else
                    {
                        double[] vals = mat.get(i, j);
                        rowValues.Add(string.Join(";", vals.Select(v => v.ToString("G3"))));
                    }
                }
                writer.WriteLine(string.Join(",", rowValues));
            }
        }
    }
    public static Color[] MatToColorArray(Mat mat)
    {
        int width = mat.cols();
        int height = mat.rows();
        int channels = mat.channels();
        byte[] data = new byte[width * height * channels];
        mat.get(0, 0, data);

        Color[] colors = new Color[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * channels;
                float b = data[idx] / 255f;
                float g = data[idx + 1] / 255f;
                float r = data[idx + 2] / 255f;
                colors[y * width + x] = new Color(r, g, b, 1f);
            }
        }
        return colors;
    }
    public void VisualizeAndSaveMatches(
    Mat img1, MatOfKeyPoint kp1,
    Mat img2, MatOfKeyPoint kp2,
    List<DMatch> matches,
    string filename = "feature_matches.png")
    {
        Mat outputMat = new Mat();
        MatOfDMatch matchesMat = new MatOfDMatch();
        matchesMat.fromList(matches);
        Mat img1Copy = img1.clone();
        Mat img2Copy = img2.clone();
        OpenCVForUnity.CoreModule.Core.flip(img1, img1Copy, 0);
        OpenCVForUnity.CoreModule.Core.flip(img2, img2Copy, 0);
        Features2d.drawMatches(
            img1Copy, kp1,
            img2Copy, kp2,
            matchesMat, outputMat,
            new Scalar(0, 255, 0),
            new Scalar(255, 0, 0),
            new MatOfByte(),
            Features2d.DrawMatchesFlags_DEFAULT);

        Texture2D outputTexture = new Texture2D(outputMat.cols(), outputMat.rows());
        Utils.matToTexture2D(outputMat, outputTexture);

        string path = System.IO.Path.Combine(Application.persistentDataPath, filename);
        File.WriteAllBytes(path, outputTexture.EncodeToPNG());
        Debug.Log($"Feature matches image saved to: {path}");
    }

    public void SaveDepthImage(Vector3[] prevDepthPoints, int width, int height, string filename = "")
    {
        float minDepth = float.MaxValue;
        float maxDepth = float.MinValue;
        for (int i = 0; i < prevDepthPoints.Length; i++)
        {
            float z = Vector3.Distance(prevDepthPoints[i], Vector3.zero);
            if (z > 0.0001f)
            {
                if (z < minDepth) minDepth = z;
                if (z > maxDepth) maxDepth = z;
            }
        }
        if (minDepth == float.MaxValue || maxDepth == float.MinValue)
        {
            Debug.LogWarning("No valid depth values found for saving depth image.");
            return;
        }

        Texture2D depthTex = new Texture2D(width, height, TextureFormat.R8, false);
        Color[] pixels = new Color[width * height];
        for (int i = 0; i < prevDepthPoints.Length; i++)
        {
            float z = Vector3.Distance(prevDepthPoints[i], Vector3.zero);
            float norm = (z > 0.0001f) ? (z - minDepth) / (maxDepth - minDepth) : 0f;
            pixels[i] = new Color(norm, norm, norm, 1f);
        }
        depthTex.SetPixels(pixels);
        depthTex.Apply();

        string path = System.IO.Path.Combine(Application.persistentDataPath, filename);
        File.WriteAllBytes(path, depthTex.EncodeToPNG());
        Debug.Log($"Depth image saved to: {path}");
    }
    public void VisualizeAndSaveDepthMatches(
    Vector3[] prevDepthPoints, Vector3[] currDepthPoints,
    int width, int height,
    List<DMatch> matches,
    MatOfKeyPoint prevKeypoints, MatOfKeyPoint currKeypoints,
    string prevFilename = "prev_depth_matches.png",
    string currFilename = "curr_depth_matches.png")
    {
        Texture2D prevDepthTex = new Texture2D(width, height, TextureFormat.RGB24, false);
        Texture2D currDepthTex = new Texture2D(width, height, TextureFormat.RGB24, false);

        float minDepth = float.MaxValue, maxDepth = float.MinValue;
        foreach (var pt in prevDepthPoints.Concat(currDepthPoints))
        {
            float z = Vector3.Distance(pt, Vector3.zero);
            if (z > 0.0001f)
            {
                if (z < minDepth) minDepth = z;
                if (z > maxDepth) maxDepth = z;
            }
        }
        if (minDepth == float.MaxValue || maxDepth == float.MinValue)
        {
            Debug.LogWarning("No valid depth values found for match visualization.");
            return;
        }


        Color[] prevPixels = new Color[width * height];
        Color[] currPixels = new Color[width * height];
        for (int i = 0; i < prevDepthPoints.Length; i++)
        {
            float z = Vector3.Distance(prevDepthPoints[i], Vector3.zero);
            float norm = (z > 0.0001f) ? (z - minDepth) / (maxDepth - minDepth) : 0f;
            prevPixels[i] = new Color(norm, norm, norm, 1f);
        }
        for (int i = 0; i < currDepthPoints.Length; i++)
        {
            float z = Vector3.Distance(currDepthPoints[i], Vector3.zero);
            float norm = (z > 0.0001f) ? (z - minDepth) / (maxDepth - minDepth) : 0f;
            currPixels[i] = new Color(norm, norm, norm, 1f);
        }
        prevDepthTex.SetPixels(prevPixels);
        prevDepthTex.Apply();
        currDepthTex.SetPixels(currPixels);
        currDepthTex.Apply();

        KeyPoint[] prevKPArray = prevKeypoints.toArray();
        KeyPoint[] currKPArray = currKeypoints.toArray();

        foreach (var match in matches)
        {
            int idxPrev = match.queryIdx;
            if (idxPrev >= 0 && idxPrev < prevKPArray.Length)
            {
                int cx = (int)prevKPArray[idxPrev].pt.x;
                int cy = (int)prevKPArray[idxPrev].pt.y;
                for (int y = -2; y <= 2; y++)
                {
                    for (int x = -2; x <= 2; x++)
                    {
                        if (x * x + y * y <= 4)
                        {
                            int px = cx + x;
                            int py = cy + y;
                            if (px >= 0 && px < width && py >= 0 && py < height)
                                prevDepthTex.SetPixel(px, py, Color.red);
                        }
                    }
                }
            }
            int idxCurr = match.trainIdx;
            if (idxCurr >= 0 && idxCurr < currKPArray.Length)
            {
                int cx = (int)currKPArray[idxCurr].pt.x;
                int cy = (int)currKPArray[idxCurr].pt.y;
                for (int y = -2; y <= 2; y++)
                {
                    for (int x = -2; x <= 2; x++)
                    {
                        if (x * x + y * y <= 4)
                        {
                            int px = cx + x;
                            int py = cy + y;
                            if (px >= 0 && px < width && py >= 0 && py < height)
                                currDepthTex.SetPixel(px, py, Color.red);
                        }
                    }
                }
            }
        }
        prevDepthTex.Apply();
        currDepthTex.Apply();


        string prevPath = System.IO.Path.Combine(Application.persistentDataPath, prevFilename);
        string currPath = System.IO.Path.Combine(Application.persistentDataPath, currFilename);
        File.WriteAllBytes(prevPath, prevDepthTex.EncodeToPNG());
        File.WriteAllBytes(currPath, currDepthTex.EncodeToPNG());
        Debug.Log($"Saved depth match images to: {prevPath} and {currPath}");
    }

    public void SaveCurrentPCD()
    {
        SaveVector3ListToPCD(RenderPointCloud, RenderPointCloudColor, Application.persistentDataPath + $"/fusedPointCloud{DateTime.Now:dd-MM-yyyy-HH-mm-ss}.pcd");
    }
    public void SaveVector3ListToPCD(List<Vector3> points, List<Color> colors, string filePath)
    {
        if (points == null || points.Count == 0)
        {
            Debug.LogWarning("Point list is empty or null");
            return;
        }


        string header = $"# .PCD v0.7 - Point Cloud Data file format\n" +
                $"VERSION 0.7\n" +
                $"FIELDS x y z rgb\n" +
                $"SIZE 4 4 4 4\n" +
                $"TYPE F F F F\n" +
                $"COUNT 1 1 1 1\n" +
                $"WIDTH {points.Count}\n" +
                $"HEIGHT 1\n" +
                $"VIEWPOINT 0 0 0 1 0 0 0\n" +
                $"POINTS {points.Count}\n" +
                $"DATA binary\n";

        File.WriteAllText(filePath, header, Encoding.ASCII);

        using (FileStream fileStream = new FileStream(filePath, FileMode.Append))
        using (BinaryWriter binaryWriter = new BinaryWriter(fileStream))
        {
            for (int i = 0; i < points.Count; i++)
            {
                Vector3 point = points[i];
                Color color = colors[i];

                binaryWriter.Write(point.x);
                binaryWriter.Write(point.y);
                binaryWriter.Write(point.z);
                int r = Mathf.Clamp((int)(color.r * 255.0f), 0, 255);
                int g = Mathf.Clamp((int)(color.g * 255.0f), 0, 255);
                int b = Mathf.Clamp((int)(color.b * 255.0f), 0, 255);
                int rgb = (r << 16) | (g << 8) | b;
                float rgbFloat = BitConverter.ToSingle(BitConverter.GetBytes(rgb), 0);

                binaryWriter.Write(rgbFloat);
            }
        }
        Debug.Log($"PCD file saved to: {filePath} with {points.Count} points");
    }
    public void DeletePointByRay(Ray ray, float radius)
    {
        if (PointCloud == null || PointCloud.Count == 0)
        {
            Debug.LogWarning("Point cloud is empty");
            return;
        }

        for (int i = PointCloud.Count - 1; i >= 0; i--)
        {
            Vector3 point = PointCloud[i];
            // Compute shortest distance from point to ray
            Vector3 toPoint = point - ray.origin;
            float t = Vector3.Dot(toPoint, ray.direction);
            Vector3 closest = ray.origin + ray.direction * t;
            float dist = Vector3.Distance(point, closest);
            if (dist < radius)
            {
                PointCloud.RemoveAt(i);
                PointCloudColor.RemoveAt(i);
            }
        }
        var keysToRemove = new List<Vector3Int>();
        foreach (var kvp in tsdfGrid)
        {
            Vector3Int voxel = kvp.Key;
            Vector3 voxelCenter = new Vector3(
                (voxel.x + 0.5f) * voxelSize,
                (voxel.y + 0.5f) * voxelSize,
                (voxel.z + 0.5f) * voxelSize
            );
            Vector3 toVoxel = voxelCenter - ray.origin;
            float t = Vector3.Dot(toVoxel, ray.direction);
            Vector3 closest = ray.origin + ray.direction * t;
            float dist = Vector3.Distance(voxelCenter, closest);
            if (dist < radius)
            {
                keysToRemove.Add(voxel);
            }
        }
        foreach (var key in keysToRemove)
        {
            tsdfGrid.Remove(key);
        }
        RenderPCD();
    }
}
