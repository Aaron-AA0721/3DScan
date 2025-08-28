using System;
using System.Buffers;
using Core;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using MagicLeap.Android;
using Unity.Collections;
using UnityEngine;
using UnityEngine.XR.OpenXR;
using MagicLeap.OpenXR.Features.PixelSensors;
using System.Text;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine.Android;
using Random = UnityEngine.Random;
using Unity.Burst.Intrinsics;
using XRUtils = Unity.XR.CoreUtils;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using Utils = OpenCVForUnity.UnityUtils.Utils;
using g4;
namespace Components
{
    public class OpenXRDataCapture : MonoBehaviour
    {
        [SerializeField] private StatusWindow progressHandler;

        private readonly List<SensorInfo> activeSensors = new();

        private readonly HashSet<string> grantedPermissions = new();
        private readonly Dictionary<PixelSensorType, List<PixelSensorId>> sensorIdTable = new();

        private List<PixelSensorId> availableSensors = new();
        SensorInfo RGBsensor;
        SensorInfo Depthsensor;

        private bool isInitialized;
        private MagicLeapPixelSensorFeature pixelSensorFeature;

        public static bool UseRawDepth = true;

        public static Vector3[] DepthPoints = new Vector3[1];
        public static Color[] RGBColors = new Color[1];
        static XRUtils.XROrigin xrOrigin;
        static Fusion fusion;

        private void Awake()
        {
            pixelSensorFeature = OpenXRSettings.Instance.GetFeature<MagicLeapPixelSensorFeature>();
            xrOrigin = FindFirstObjectByType<XRUtils.XROrigin>();
            fusion = FindFirstObjectByType<Fusion>();
            if (xrOrigin != null) Debug.Log("found XROrigin");
            if (pixelSensorFeature == null || !pixelSensorFeature.enabled)
            {
                enabled = false;
                progressHandler.Fail(
                    "PixelSensorFeature is either not enabled or is null. Check Project Settings in the Unity editor to make sure the feature is enabled");

                grantedPermissions.Clear();
                Permissions.RequestPermission(Permission.Camera, OnPermissionGranted, null,
                    null);
            }
        }


        private void Update()
        {
            if (!isInitialized)
            {
                Initialize();
            }

        }
        public void InitializeCameras()
        {
            InitializeRGBCamera();
            InitializeDepthCamera();
        }
        public void InitializeDepthCamera()
        {
            var newSensorType = PixelSensorType.Depth;
            progressHandler.Log("Initializing Depth Camera...");
            StartCoroutine(CreateSensorAfterPermission(newSensorType));
        }

        public void InitializeRGBCamera()
        {
            var newSensorType = PixelSensorType.Picture;
            progressHandler.Log("Initializing Picture Camera...");
            StartCoroutine(CreateSensorAfterPermission(newSensorType));
        }
        public void ClearPCD()
        {
            Array.Clear(DepthPoints,0,DepthPoints.Length);
            Array.Clear(RGBColors,0,RGBColors.Length);
            PointCloudRenderer PCD_Renderer = GameObject.FindObjectOfType<PointCloudRenderer>();
            PCD_Renderer.Clear();
        }
        public Texture2D CaptureDepthImage(out Texture2D confidenceTexture)
        {
            progressHandler.Log("Capturing Depth Image...");
            var frameDataText = new StringBuilder();
            if (Depthsensor == null)
            {
                foreach (var sensor in activeSensors)
                {
                    sensorIdTable.TryGetValue(PixelSensorType.Depth, out var sensors);
                    if (sensors != null && sensors.Contains(sensor.SensorId))
                    {
                        Depthsensor = sensor;
                    }
                }
            }
            Texture2D texture = Depthsensor.Capture(frameDataText, out confidenceTexture);
            progressHandler.Log(frameDataText.ToString());
            return texture;
        }
        public Texture2D CaptureRGBImage()
        {
            progressHandler.Log("Capturing RGB Image...");
            var frameDataText = new StringBuilder();
            if (RGBsensor == null)
            {
                foreach (var sensor in activeSensors)
                {
                    sensorIdTable.TryGetValue(PixelSensorType.Picture, out var sensors);
                    if (sensors != null && sensors.Contains(sensor.SensorId))
                    {
                        RGBsensor = sensor;
                    }
                }
            }

            Texture2D texture = RGBsensor.Capture(frameDataText, out var confidenceTexture);
            progressHandler.Log(frameDataText.ToString());
            return texture;
        }
        


        void AddToTable(PixelSensorType key, PixelSensorId sensorId)
        {
            if (sensorIdTable.TryGetValue(key, out var sensorList))
            {
                sensorList.Add(sensorId);
            }
            else
            {
                sensorIdTable.Add(key, new List<PixelSensorId>
                {
                    sensorId
                });
            }
        }

        private void Initialize()
        {


            availableSensors = pixelSensorFeature.GetSupportedSensors();


            foreach (var sensor in availableSensors)
            {
                if (sensor.SensorName.Contains("World"))
                {
                    AddToTable(PixelSensorType.World, sensor);
                }

                if (sensor.SensorName.Contains("Eye"))
                {
                    AddToTable(PixelSensorType.Eye, sensor);
                }

                if (sensor.SensorName.Contains("Picture"))
                {
                    AddToTable(PixelSensorType.Picture, sensor);
                }

                if (sensor.SensorName.Contains("Depth"))
                {
                    AddToTable(PixelSensorType.Depth, sensor);
                }
            }

            isInitialized = true;
        }

        private void OnPermissionGranted(string permission)
        {
            grantedPermissions.Add(permission);
        }

        private void StopActiveSensors()
        {
            foreach (var sensor in activeSensors)
            {
                sensor.Dispose();
            }

            activeSensors.Clear();
        }

        private void OnDestroy()
        {
            StopActiveSensors();
        }


        private string NeededPermission(PixelSensorType sensorType)
        {
            var result = sensorType switch
            {
                PixelSensorType.Depth => Permissions.DepthCamera,
                PixelSensorType.World or PixelSensorType.Picture => Permission.Camera,
                PixelSensorType.Eye => Permissions.EyeCamera,
                _ => ""
            };
            return result;
        }



        public List<Vector3> Convert_PCD_To_VectorList(string filePath)
        {
            List<Vector3> points = new List<Vector3>();

            if (!File.Exists(filePath))
            {
                progressHandler.Log("File not found: " + filePath);
                return points;
            }
            try
            {
                FileStream stream = new FileStream(filePath, FileMode.Open);
                BinaryReader reader = new BinaryReader(stream);
                bool inHeader = true;
                int pointsCount = 0;
                int dataStartLine = 0;
                List<string> fields = new List<string>();
                List<int> sizes = new List<int>();
                bool hasX = false, hasY = false, hasZ = false;
                int xIndex = -1, yIndex = -1, zIndex = -1, rgbIndex = -1;
                string dataType = "";
                progressHandler.Log($"PCD Reading {filePath}");
                string line = "";

                while (inHeader && stream.Position < stream.Length)
                {
                    line = ReadPcdLine(reader);
                    if (line.StartsWith("#") || string.IsNullOrWhiteSpace(line))
                        continue;

                    if (line.StartsWith("DATA"))
                    {
                        dataType = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)[1].ToLower();
                        inHeader = false;
                        break;
                    }

                    if (line.StartsWith("FIELDS"))
                    {
                        string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        for (int j = 1; j < parts.Length; j++)
                        {
                            fields.Add(parts[j].ToLower());
                            if (parts[j].ToLower() == "x") { hasX = true; xIndex = j - 1; }
                            if (parts[j].ToLower() == "y") { hasY = true; yIndex = j - 1; }
                            if (parts[j].ToLower() == "z") { hasZ = true; zIndex = j - 1; }
                            if (parts[j].ToLower() == "rgb") rgbIndex = j - 1;
                        }
                    }
                    if (line.StartsWith("SIZE"))
                    {
                        string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        for (int j = 1; j < parts.Length; j++)
                        {
                            if (int.TryParse(parts[j], out int size))
                                sizes.Add(size);
                        }
                    }
                    else if (line.StartsWith("POINTS"))
                    {
                        string[] parts = line.Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length > 1)
                            int.TryParse(parts[1], out pointsCount);
                    }
                }
                if (!hasX || !hasY || !hasZ)
                {
                    progressHandler.Log("PCD file doesn't contain x, y, z coordinates");
                    return points;
                }




                int pointByteSize = 0;
                for (int i = 0; i < sizes.Count; i++)
                    pointByteSize += sizes[i];
                progressHandler.Log($"point count: {pointsCount}, pointByteSize: {pointByteSize}");
                progressHandler.Log($"dataType: {dataType}, ");
                long dataStartPos = stream.Position;
                long totalBytes = stream.Length - dataStartPos;
                int estimatedPoints = (int)(totalBytes / pointByteSize);
                progressHandler.Log($"estimatedPoints: {estimatedPoints}");
                for (int i = 0; i < estimatedPoints; i++)
                {
                    if (stream.Position + pointByteSize > stream.Length)
                        break;

                    float x = 0, y = 0, z = 0, rgb = 0;

                    for (int fieldIdx = 0; fieldIdx < fields.Count; fieldIdx++)
                    {
                       
                        if (sizes[fieldIdx] == 4) 
                        {
                            float value = reader.ReadSingle();

                            if (fieldIdx == xIndex) x = value;
                            else if (fieldIdx == yIndex) y = value;
                            else if (fieldIdx == zIndex) z = value;
                            else if (fieldIdx == rgbIndex) rgb = value;
                        }

                        else
                        {
                            reader.BaseStream.Seek(sizes[fieldIdx], SeekOrigin.Current);
                            progressHandler.Log($"Size Not supported: {sizes[fieldIdx]}");
                        }
                    }
                    points.Add(new Vector3(x, y, z));
                }

                progressHandler.Log($"Loaded {points.Count} points from PCD file");
                return points;
            }
            catch (Exception ex)
            {
                progressHandler.Log($"Error processing PCD file: {ex.Message}");
                points = new List<Vector3>();
            }
            return points;
        }
        private string ReadPcdLine(BinaryReader reader)
        {
            List<byte> bytes = new List<byte>();
            byte b;

            while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                b = reader.ReadByte();
                if (b == '\n') break;
                if (b != '\r') bytes.Add(b);
            }

            return Encoding.ASCII.GetString(bytes.ToArray());
        }
        public void RenderPCD(List<Vector3> pointCloud, List<Color> colors = null)
        {
            PointCloudRenderer PCD_Renderer = GameObject.FindObjectOfType<PointCloudRenderer>();
            progressHandler.Log("PCD Rendering");
            Debug.Log("PCD Rendering: " + pointCloud.Count + "|" + colors.Count);
            if (colors == null)
            {
                Debug.Log("CREATE colors because null");
                colors = new List<Color>();
                for (int i = 0; i < pointCloud.Count; i++)
                {
                    colors.Add(new Color(0.5f, 0.5f, 0.5f, 1));
                }
            }

            PCD_Renderer.Render(pointCloud.ToArray(), colors.ToArray());
        }
        public void AppendRenderPCD(List<Vector3> pointCloud, List<Color> colors = null)
        {
            PointCloudRenderer PCD_Renderer = GameObject.FindObjectOfType<PointCloudRenderer>();
            progressHandler.Log("PCD Rendering");
            Debug.Log("PCD Rendering: " + pointCloud.Count + "|" + colors.Count);
            if (colors == null)
            {
                Debug.Log("CREATE colors because null");
                colors = new List<Color>();
                for (int i = 0; i < pointCloud.Count; i++)
                {
                    colors.Add(new Color(0.5f, 0.5f, 0.5f, 1));
                }
            }

            PCD_Renderer.AppendRender(pointCloud.ToArray(), colors.ToArray());
        }
        public void CaptureRGBDImage()
        {
            var rgbTexture = CaptureRGBImage();
            var depthTexture = CaptureDepthImage(out var confidenceTexture);
            Array.Clear(DepthPoints,0,DepthPoints.Length);
            Array.Clear(RGBColors,0,RGBColors.Length);
            ProcessDepthPoints(ref DepthPoints, ref RGBColors, depthTexture, confidenceTexture, rgbTexture,out var undistortedDepth, out var undistortedRGB);
            fusion.PCDFusion(DepthPoints, RGBColors, undistortedDepth, undistortedRGB, Depthsensor.sensorPose, RGBsensor.sensorPose);
        }
        private void ProcessDepthPoints(ref Vector3[] depthPointArray, ref Color[] rgbColors, Texture2D depthTexture, Texture2D confidenceTexture, Texture2D rgbTexture, out Texture2D undistortedDepth, out Texture2D undistortedRGB)
        {
            Vector2Int depthResolution = new Vector2Int(depthTexture.width, depthTexture.height);
            Debug.Log($"DepthCloud: Processing Depth. Resolution : {depthResolution.x} x {depthResolution.y}");
            Texture2D UndistortedConfidence = UndistortDepth(confidenceTexture);
            undistortedDepth = UndistortDepth(depthTexture);
            undistortedRGB = rgbTexture;
            //undistortedRGB = UndistortRGB(rgbTexture);
            //wtf??? without undistortion the coloring looks better?
            //TODO: check magic leap intrinsics

            NativeArray<float> depthData = undistortedDepth.GetRawTextureData<float>();
            NativeArray<float> confidenceData = UndistortedConfidence.GetRawTextureData<float>();
            var DepthCameraToWorldMatrix = Matrix4x4.TRS(Depthsensor.sensorPose.position, Depthsensor.sensorPose.rotation, Vector3.one);
            var RGBWorldToCameraMatrix = Matrix4x4.TRS(RGBsensor.sensorPose.position, RGBsensor.sensorPose.rotation, Vector3.one).inverse;
            double[] RGBRadicalDistortion;
            double[] RGBTangentialDistortion;
            Vector2 RGBFocalLength = new Vector2();
            Vector2 RGBPrincipalPoint = new Vector2();
            Color[] rgbPixels = undistortedRGB.GetPixels();
            rgbColors = rgbPixels;
            depthPointArray = new Vector3[rgbPixels.Length];
            Array.Fill(depthPointArray, Vector3.zero);
            if (RGBsensor.fisheyeIntrinsics == null)
            {
                Debug.Log("RGB intrinsics null");
                RGBRadicalDistortion = new double[4] { 0.1, -0.4, 0.4, 0.0 };
                RGBTangentialDistortion = new double[2] { 0, 0 };
                RGBFocalLength = new Vector2(503.30f, 503.31f);
                RGBPrincipalPoint = new Vector2(320.40f, 242.70f);
            }
            else
            {
                RGBRadicalDistortion = RGBsensor.fisheyeIntrinsics.RadialDistortion;
                RGBTangentialDistortion = RGBsensor.fisheyeIntrinsics.TangentialDistortion;
                RGBPrincipalPoint = RGBsensor.fisheyeIntrinsics.PrincipalPoint;
                RGBFocalLength = RGBsensor.fisheyeIntrinsics.FocalLength;
            }
            for (int y = 0; y < depthResolution.y; ++y)
            {
                for (int x = 0; x < depthResolution.x; ++x)
                {
                    
                    int index = x + (depthResolution.y - y - 1) * depthResolution.x;
                    float depth = depthData[index];

                    if (confidenceData[index] <= -0.091f)
                    {
                        continue;
                    }

                    Vector2 uv = Depthsensor.cachedProjectionTable[y, x];
                    Vector3 cameraPoint = new Vector3(uv.x, uv.y, 1).normalized * depth;
                    Vector3 worldPoint = DepthCameraToWorldMatrix.MultiplyPoint3x4(cameraPoint);
                    Vector3 RGBcameraPoint = RGBWorldToCameraMatrix.MultiplyPoint3x4(worldPoint);
                    float xx = RGBFocalLength.x * (RGBcameraPoint.x / RGBcameraPoint.z) + RGBPrincipalPoint.x;
                    float yy = RGBFocalLength.y * (RGBcameraPoint.y / RGBcameraPoint.z) + RGBPrincipalPoint.y;
                    if (xx >= 0 && xx < RGBsensor.frame_resolution.x && yy >= 0 && yy < RGBsensor.frame_resolution.y)
                    {
                        
                        try
                        {
                            depthPointArray[(int)yy * RGBsensor.frame_resolution.x + (int)xx] = cameraPoint;
                        }
                        catch (System.Exception e)
                        {
                            Debug.Log(xx + "|" + yy + "|" + RGBsensor.frame_resolution.x + "|" + RGBsensor.frame_resolution.y);
                            Debug.LogError("Error assigning depth point: " + e.Message);

                        }

                      
                    }
                }
            }
            Destroy(UndistortedConfidence);
            depthData.Dispose();
            confidenceData.Dispose();
        }
        Texture2D UndistortDepth(Texture2D depthTexture) {
            Texture2D texture = depthTexture;
            Mat depth_Mat = new Mat(depthTexture.height, depthTexture.width, CvType.CV_32FC1);
            NativeArray<float> depthData = depthTexture.GetRawTextureData<float>();
            depth_Mat.put(0, 0, depthData.ToArray());
    
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
            Mat distCoeffs = new Mat(1, 5, CvType.CV_64FC1);

            PixelSensorPinholeIntrinsics intrinsics = Depthsensor.pinholeIntrinsics;
            double[] distortionArray = new double[5];
            Vector2 FocalLength = new Vector2();
            Vector2 PrincipalPoint = new Vector2();
            if (intrinsics == null)
            {
                Debug.Log("intrinsics null");
                distortionArray = new double[5] { -0.091, -0.013, 0, 0, -0.011 };
                FocalLength = new Vector2(363.11f, 363.11f);
                PrincipalPoint = new Vector2(267.86f, 237.83f);
            }
            else
            {
                distortionArray = intrinsics.Distortion;
                PrincipalPoint = intrinsics.PrincipalPoint;
                FocalLength = intrinsics.FocalLength;
            }
        
            
            cameraMatrix.put(0, 0, FocalLength.x, 0, PrincipalPoint.x);
            cameraMatrix.put(1, 0, 0, FocalLength.y, PrincipalPoint.y);
            cameraMatrix.put(2, 0, 0, 0, 1);
            
            distCoeffs.put(0, 0, distortionArray);
            
            Mat undistorted_Mat = new Mat(depthTexture.height, depthTexture.width, CvType.CV_32FC1);
            
            Calib3d.undistort(depth_Mat, undistorted_Mat, cameraMatrix, distCoeffs);
            
            float[] floatArray = new float[undistorted_Mat.cols() * undistorted_Mat.rows()];

            undistorted_Mat.get(0, 0, floatArray);
            
            byte[] byteArray = new byte[floatArray.Length * sizeof(float)];
            Buffer.BlockCopy(floatArray, 0, byteArray, 0, byteArray.Length);
            
            texture.LoadRawTextureData(byteArray);
            texture.Apply();
            
            depth_Mat.Dispose();
            cameraMatrix.Dispose();
            distCoeffs.Dispose();
            undistorted_Mat.Dispose();
            return texture;
        }
        Texture2D UndistortRGB(Texture2D rgbTexture) {
            Texture2D texture = new Texture2D(rgbTexture.width, rgbTexture.height);
            Mat RGB_Mat = new Mat(rgbTexture.height, rgbTexture.width, CvType.CV_8UC3);
            Utils.texture2DToMat(rgbTexture, RGB_Mat);
           
            Mat BGR_Mat = new Mat(rgbTexture.height, rgbTexture.width, CvType.CV_8UC3);
            Imgproc.cvtColor(RGB_Mat, BGR_Mat, Imgproc.COLOR_RGB2BGR);
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
            Mat distCoeffs = new Mat(1, 4, CvType.CV_64FC1);
            PixelSensorFisheyeIntrinsics intrinsics = RGBsensor.fisheyeIntrinsics;
            double[] RGBRadicalDistortion;
            double[] RGBTangentialDistortion;
            Vector2 RGBFocalLength = new Vector2();
            Vector2 RGBPrincipalPoint = new Vector2();
            if (intrinsics == null)
            {
                Debug.Log("RGB intrinsics null");
                RGBRadicalDistortion = new double[4] { 0.1, -0.4, 0.4, 0.0};
                RGBTangentialDistortion = new double[2] { 0, 0 };
                RGBFocalLength = new Vector2(503.30f, 503.31f);
                RGBPrincipalPoint = new Vector2(320.40f, 242.70f);
            }
            else
            {
                RGBRadicalDistortion = new double[4];
                Array.Copy(RGBsensor.fisheyeIntrinsics.RadialDistortion, RGBRadicalDistortion, 4);
                RGBTangentialDistortion = RGBsensor.fisheyeIntrinsics.TangentialDistortion;
                RGBPrincipalPoint = RGBsensor.fisheyeIntrinsics.PrincipalPoint;
                RGBFocalLength = RGBsensor.fisheyeIntrinsics.FocalLength;
            }
            cameraMatrix.put(0, 0, RGBFocalLength.x, 0, RGBPrincipalPoint.x);
            cameraMatrix.put(1, 0, 0, RGBFocalLength.y, RGBPrincipalPoint.y);
            cameraMatrix.put(2, 0, 0, 0, 1);

           
            distCoeffs.put(0, 0, RGBRadicalDistortion);
            Mat undistorted_Mat = new Mat();

            Debug.Log(cameraMatrix.dump()+"|\n"+distCoeffs.dump());
            Size size = new Size(rgbTexture.width, rgbTexture.height);
            Calib3d.fisheye_undistortImage(BGR_Mat, undistorted_Mat, cameraMatrix, distCoeffs, cameraMatrix, size);
            Mat RGB_undistorted_Mat = new Mat(undistorted_Mat.height(), undistorted_Mat.width(), CvType.CV_8UC3);
            Imgproc.cvtColor(undistorted_Mat, RGB_undistorted_Mat, Imgproc.COLOR_BGR2RGB);
            Utils.matToTexture2D(RGB_undistorted_Mat, texture);
            
            return texture;
        }
        private IEnumerator CreateSensorAfterPermission(PixelSensorType newSensorType)
        {
            var neededPermission = NeededPermission(newSensorType);
            if (!grantedPermissions.Contains(neededPermission))
            {
                Permissions.RequestPermission(neededPermission, OnPermissionGranted, null);
            }

            yield return new WaitUntil(() => grantedPermissions.Contains(neededPermission));
           
            if (!sensorIdTable.TryGetValue(newSensorType, out var sensors))
            {
                progressHandler.Fail($"The given {newSensorType} does not have any associated sensors");
                yield break;
            }

            foreach (var sensor in sensors)
            {
                var sensorInfo = new SensorInfo(pixelSensorFeature, sensor);
                

                if (!sensorInfo.CreateSensor())
                {
                    progressHandler.Fail($"Unable to create sensor type: {sensorInfo.SensorId.SensorName}");
                    yield break;

                }

                activeSensors.Add(sensorInfo);

                sensorInfo.Created = true;
                sensorInfo.Initialize();


                if (!sensorInfo.Created)
                {
                    yield break;
                }

                var configureResult = sensorInfo.ConfigureSensorRoutine();
                yield return configureResult;
                if (!configureResult.DidOperationSucceed)
                {
                    progressHandler.Fail($"Unable to configure {sensorInfo.SensorId.SensorName}");
                    yield break;
                }

                sensorInfo.Configured = true;


                if (!sensorInfo.Configured)
                {
                    yield break;
                }

                var startSensorResult = sensorInfo.StartSensorRoutine();
                yield return startSensorResult;
                if (!startSensorResult.DidOperationSucceed)
                {
                    yield break;
                }

                sensorInfo.Started = true;

                yield return new WaitForSeconds(2);
                if (sensorInfo.Started)
                {
                    sensorInfo.ShouldFetchData = true;
                }

                progressHandler.Log($"Sensor {sensorInfo.SensorId.SensorName} created and configured.");
                if (sensorInfo.SensorId.SensorName.Contains("Depths"))
                {
                    Depthsensor = sensorInfo;
                }
                if (sensorInfo.SensorId.SensorName.Contains("RGB"))
                {
                    RGBsensor = sensorInfo;
                }
            }
        }
        class SensorInfo : IDisposable
        {
            public SensorInfo(MagicLeapPixelSensorFeature feature, PixelSensorId sensorId)
            {
                PixelSensorFeature = feature;
                SensorId = sensorId;
                ConfiguredStream = 0;
            }

            public PixelSensorId SensorId { get; }

            private uint ConfiguredStream { get; }

            private MagicLeapPixelSensorFeature PixelSensorFeature { get; }

            public bool Configured { get; set; }

            public bool Started { get; set; }

            public bool Created { get; set; }

            public bool ShouldFetchData { get; set; }
            public List<List<string>> fileBatchs = new List<List<string>>();
            public Vector2[,] cachedProjectionTable { get; private set; }
            public Pose sensorPose { get; private set; }
            public Vector2Int frame_resolution { get; private set; } = new Vector2Int(0, 0);


            public PixelSensorFisheyeIntrinsics fisheyeIntrinsics { get; private set; }
            public PixelSensorPinholeIntrinsics pinholeIntrinsics { get; private set; }
            private PixelSensorCapabilityType[] targetCapabilityTypes = new[]
            {
                    PixelSensorCapabilityType.UpdateRate,
                    PixelSensorCapabilityType.Format,
                    PixelSensorCapabilityType.Resolution,
                    PixelSensorCapabilityType.Depth,
                };

            public void Dispose()
            {
                if (!Created)
                {
                    return;
                }

                PixelSensorFeature.DestroyPixelSensor(SensorId);
            }


            public Texture2D Capture(StringBuilder frameDataText, out Texture2D confidenceTexture)
            {
                Texture2D texture = null;
                confidenceTexture = null;
                PixelSensorPinholeIntrinsics pinhole = null;
                PixelSensorFisheyeIntrinsics fisheye = null;
                if (!PixelSensorFeature.GetSensorData(SensorId, ConfiguredStream, out PixelSensorFrame frame, out PixelSensorMetaData[] metaData, Allocator.Temp, shouldFlipTexture: false))
                {
                    return texture;
                }
                frameDataText.AppendLine("[Metadata]");
                frameDataText.AppendLine($"Capture Time: {frame.CaptureTime}");
                frameDataText.AppendLine($"Sensor: {SensorId.SensorName}");
                frameDataText.AppendLine($"Frame Type: {frame.FrameType}");
                frameDataText.AppendLine($"Frame Valid: {frame.IsValid}");
                frameDataText.AppendLine($"Frame Plane Count: {frame.Planes.Length}");
                GetStringFromMetaData(in metaData, in frameDataText, out pinhole, out fisheye);
                pinholeIntrinsics = pinhole;
                fisheyeIntrinsics = fisheye;
                frameDataText.AppendLine($"[Extrinsics]");
                var frameRotation = PixelSensorFeature.GetSensorFrameRotation(SensorId);
                if (xrOrigin == null) xrOrigin = FindFirstObjectByType<XRUtils.XROrigin>();
                if (xrOrigin != null)
                {
                    Pose offset = new Pose(
                    xrOrigin.CameraFloorOffsetObject.transform.position,
                    xrOrigin.transform.rotation);
                    sensorPose = PixelSensorFeature.GetSensorPose(SensorId, frame.CaptureTime, offset);
                    //reference:https://forum.magicleap.cloud/t/hologram-drift-issue-when-tracking-object-with-retroreflective-markers-using-depth-camera-raw-data/5618
                }
                else
                {
                    sensorPose = PixelSensorFeature.GetSensorPose(SensorId);
                }
                frameDataText.AppendLine($"Sensor Pose Position: {sensorPose.position}");
                frameDataText.AppendLine($"Sensor Pose Rotation: {sensorPose.rotation}");
                frameDataText.AppendLine($"Frame Rotation: {frameRotation}");

                var confidenceBuffer = metaData.OfType<PixelSensorDepthConfidenceBuffer>().FirstOrDefault();
                var flagBuffer = metaData.OfType<PixelSensorDepthFlagBuffer>().FirstOrDefault();
                uint height, width;
                Debug.Log("ProcessFrame");
                texture = ProcessFrame(in frame, in frameDataText, in confidenceBuffer, in flagBuffer, out height, out width);
                Debug.Log("resolution: " + height + " | " + width);
                frame_resolution = new Vector2Int((int)width, (int)height);

                if (frame.FrameType == PixelSensorFrameType.Depth32)
                {
                    Debug.Log("ProcessDepthConfidenceData");
                    confidenceTexture = ProcessDepthConfidenceData(in confidenceBuffer, in height, in width);

                    Debug.Log("pinholeIntrinsics= " + pinholeIntrinsics);
                    if (cachedProjectionTable == null)
                        cachedProjectionTable = CreateDepthProjectionTable(pinholeIntrinsics, frame_resolution);


                }
                if (frame.FrameType == PixelSensorFrameType.Jpeg)
                {

                    Debug.Log("fisheyeIntrinsics= " + fisheyeIntrinsics);

                }


                return texture;
            }
            


            public Texture2D ProcessDepthConfidenceData(in PixelSensorDepthConfidenceBuffer confidenceBuffer, in uint height, in uint width)
            {
                var frame = confidenceBuffer.Frame;
                if (!frame.IsValid || frame.Planes.Length == 0)
                {
                    return null;
                }
                var texture = new Texture2D((int)width, (int)height, TextureFormat.RFloat, false);
                texture.LoadRawTextureData(frame.Planes[0].ByteData);
                texture.Apply();

                return texture;
            }
            public Vector2 UndistortPinHole(Vector2 uv, double[] distortionArray)
            {
                Vector2 Half2 = new Vector2(0.5f, 0.5f);
                Vector2 offsetFromCenter = uv - Half2;

                (double K1, double K2, double K3, double P1, double P2) distortionParameters = (distortionArray[0], distortionArray[1], distortionArray[4], distortionArray[2], distortionArray[3]);


                float rSquared = Vector2.Dot(offsetFromCenter, offsetFromCenter);
                float rSquaredSquared = rSquared * rSquared;
                float rSquaredCubed = rSquaredSquared * rSquared;

                
                Vector2 radialDistortionCorrection = offsetFromCenter * (float)(1 + distortionParameters.K1 * rSquared + distortionParameters.K2 * rSquaredSquared + distortionParameters.K3 * rSquaredCubed);

                float tangentialDistortionCorrectionX = (float)((2 * distortionParameters.P1 * offsetFromCenter.x * offsetFromCenter.y) + (distortionParameters.P2 * (rSquared + 2 * offsetFromCenter.x * offsetFromCenter.x)));
                float tangentialDistortionCorrectionY = (float)((2 * distortionParameters.P2 * offsetFromCenter.x * offsetFromCenter.y) + (distortionParameters.P1 * (rSquared + 2 * offsetFromCenter.y * offsetFromCenter.y)));
                Vector2 tangentialDistortionCorrection = new Vector2(tangentialDistortionCorrectionX, tangentialDistortionCorrectionY);

                return radialDistortionCorrection + tangentialDistortionCorrection + Half2;
            }
            private Vector2[,] CreateDepthProjectionTable(PixelSensorPinholeIntrinsics intrinsics, Vector2Int resolution)
            {
                double[] distortionArray = new double[5];
                Vector2 FocalLength = new Vector2();
                Vector2 PrincipalPoint = new Vector2();
                if (intrinsics == null)
                {
                    Debug.Log("intrinsics null");
                    distortionArray = new double[5] { -0.091, -0.013, 0, 0, -0.011 };
                    FocalLength = new Vector2(363.11f, 363.11f);
                    PrincipalPoint = new Vector2(267.86f, 237.83f);
                }
                else
                {
                    distortionArray = intrinsics.Distortion;
                    PrincipalPoint = intrinsics.PrincipalPoint;
                    FocalLength = intrinsics.FocalLength;
                }
                Debug.Log("CreateProjectionTable");
                Debug.Log("resolution: " + resolution);
                Vector2[,] projectionTable = new Vector2[resolution.y, resolution.x];
                for (int y = 0; y < resolution.y; ++y)
                {
                    for (int x = 0; x < resolution.x; ++x)
                    {
                        Vector2 uv = new Vector2(x, y) / new Vector2(resolution.x - 1, resolution.y - 1);
                        Vector2 correctedUV = UndistortPinHole(uv, distortionArray);
                        projectionTable[y, x] = ((correctedUV * new Vector2(resolution.x - 1, resolution.y - 1)) - PrincipalPoint) / FocalLength;
                    }
                }
                return projectionTable;
            }

            public Texture2D ProcessFrame(in PixelSensorFrame frame, in StringBuilder frameDataText, in PixelSensorDepthConfidenceBuffer confidenceBuffer, in PixelSensorDepthFlagBuffer flagBuffer, out uint height, out uint width)
            {
                height = 0;
                width = 0;
                List<string> filePaths = new List<string>();
                string datetime = DateTime.Now.ToString("yyyy-MM-dd'T'HH:mm:ss");
                frameDataText.AppendLine("[Frame Data]");
                frameDataText.AppendLine($"System Date and Time: {datetime}");
                string path = Path.Combine(Application.persistentDataPath, $"{datetime}-{SensorId.SensorName.Replace(" ", "")}");
                Texture2D texture = null;
                if (!frame.IsValid || frame.Planes.Length == 0)
                {
                    return texture;
                }
                var frameType = frame.FrameType;
                ref var firstPlane = ref frame.Planes[0];


                if (frame.FrameType == PixelSensorFrameType.Depth32 || frame.FrameType == PixelSensorFrameType.DepthRaw)
                {
                    width = frame.Planes[0].Width;
                    height = frame.Planes[0].Height;
                    texture = new Texture2D((int)width, (int)height, TextureFormat.RFloat, false);
                    texture.LoadRawTextureData(firstPlane.ByteData);
                    texture.Apply();

                    var byteArray = ArrayPool<byte>.Shared.Rent(firstPlane.ByteData.Length);
                    firstPlane.ByteData.CopyTo(byteArray);
                    filePaths.Add($"{path}.bin");
                    var filePath = $"{path}.bin";
                    File.WriteAllBytes(filePath, byteArray);
                    ArrayPool<byte>.Shared.Return(byteArray);
                    frameDataText.AppendLine($"File Path: {filePath}"); ;

                }
                else
                {
                    switch (frame.FrameType)
                    {
                        case PixelSensorFrameType.Jpeg:
                            {
                                var byteArray = ArrayPool<byte>.Shared.Rent(firstPlane.ByteData.Length);
                                firstPlane.ByteData.CopyTo(byteArray);
                                width = frame.Planes[0].Width;
                                height = frame.Planes[0].Height;
                                texture = new Texture2D((int)width, (int)height, TextureFormat.RGB24, false);
                                texture.LoadImage(firstPlane.ByteData.ToArray());
                                filePaths.Add($"{path}.jpg");
                                var filePath = $"{path}.jpg";
                                File.WriteAllBytes(filePath, byteArray);
                                ArrayPool<byte>.Shared.Return(byteArray);
                                frameDataText.AppendLine($"File Path: {filePath}");
                                break;
                            }

                        case PixelSensorFrameType.Yuv420888:
                            {
                                for (var i = 0; i < frame.Planes.Length; i++)
                                {
                                    ref var plane = ref frame.Planes[i];
                                    var byteArray = plane.ByteData;

                                    filePaths.Add($"{path}{i}.bin");
                                    var filePath = $"{path}{i}.bin";
                                    File.WriteAllBytes(filePath, byteArray.ToArray());
                                    frameDataText.AppendLine($"File Path: {filePath}");
                                }
                                break;
                            }

                        default:
                            Debug.LogWarning($"Unhandled frame type: {frame.FrameType}");
                            break;
                    }
                }
                var frameDataFilePath = Path.Combine(Application.persistentDataPath, $"{path}.txt");
                filePaths.Add(frameDataFilePath);
                File.WriteAllText(frameDataFilePath, frameDataText.ToString());
                fileBatchs.Add(filePaths);
                return texture;
            }

            public bool CreateSensor()
            {
                return PixelSensorFeature.CreatePixelSensor(SensorId);
            }
            public enum ShortRangeUpdateRate
            {
                FiveFps = 5, ThirtyFps = 30, SixtyFps = 60
            }

            public PixelSensorAsyncOperationResult ConfigureSensorRoutine()
            {

                PixelSensorFeature.GetPixelSensorCapabilities(SensorId, ConfiguredStream, out var capabilities);
                if (SensorId.SensorName.Contains("Depth"))
                {
                    foreach (var pixelSensorCapability in capabilities)
                    {
                        if (!targetCapabilityTypes.Contains(pixelSensorCapability.CapabilityType))
                        {
                            continue;
                        }

                        if (PixelSensorFeature.QueryPixelSensorCapability(SensorId, pixelSensorCapability.CapabilityType, ConfiguredStream, out PixelSensorCapabilityRange range) && range.IsValid)
                        {
                            if (range.CapabilityType == PixelSensorCapabilityType.UpdateRate)
                            {
                                var configData = new PixelSensorConfigData(range.CapabilityType, ConfiguredStream);
                                configData.IntValue = (uint)ShortRangeUpdateRate.FiveFps;
                                PixelSensorFeature.ApplySensorConfig(SensorId, configData);
                            }
                            else if (range.CapabilityType == PixelSensorCapabilityType.Format)
                            {
                                var configData = new PixelSensorConfigData(range.CapabilityType, ConfiguredStream);
                                configData.IntValue = (uint)range.FrameFormats[OpenXRDataCapture.UseRawDepth ? 0 : 1];
                                //0 for depth and 1 for raw depth
                                PixelSensorFeature.ApplySensorConfig(SensorId, configData);
                            }
                            else if (range.CapabilityType == PixelSensorCapabilityType.Resolution)
                            {
                                var configData = new PixelSensorConfigData(range.CapabilityType, ConfiguredStream);
                                configData.VectorValue = range.ExtentValues[0];
                                PixelSensorFeature.ApplySensorConfig(SensorId, configData);
                            }
                            else if (range.CapabilityType == PixelSensorCapabilityType.Depth)
                            {
                                var configData = new PixelSensorConfigData(range.CapabilityType, ConfiguredStream);
                                configData.FloatValue = 1.0f;
                                PixelSensorFeature.ApplySensorConfig(SensorId, configData);
                            }
                        }
                    }
                    List<uint> ConfiguredStreams = new List<uint>() { ConfiguredStream };
                    return PixelSensorFeature.ConfigureSensor(SensorId, ConfiguredStreams.ToArray());
                }
                else
                {
                    return PixelSensorFeature.ConfigureSensorWithDefaultCapabilities(SensorId, ConfiguredStream);
                }





            }

            public PixelSensorAsyncOperationResult StartSensorRoutine()
            {
                Dictionary<uint, PixelSensorMetaDataType[]> supportedMetadataTypes =
                    new Dictionary<uint, PixelSensorMetaDataType[]>();

                var requestedMetaData = new List<PixelSensorMetaDataType>
                    {
                        PixelSensorMetaDataType.DepthFlagBuffer,
                        PixelSensorMetaDataType.DepthConfidenceBuffer,
                        PixelSensorMetaDataType.AnalogGain,
                        PixelSensorMetaDataType.ExposureTime,
                        PixelSensorMetaDataType.DigitalGain,
                        PixelSensorMetaDataType.FishEyeCameraModel,
                        PixelSensorMetaDataType.PinholeCameraModel
                    };

                if (PixelSensorFeature.EnumeratePixelSensorMetaDataTypes(SensorId, ConfiguredStream, out var availableMetaTypes))
                {
                    var filteredMeta = requestedMetaData
                        .Where(type => availableMetaTypes.Contains(type))
                        .ToArray();

                    supportedMetadataTypes[ConfiguredStream] = filteredMeta;

                    Debug.Log($"Metadata types for stream {ConfiguredStream} set to: {string.Join(", ", filteredMeta)}");
                }
                else
                {
                    Debug.LogError($"Failed to retrieve metadata types for stream {ConfiguredStream}. Using requested list.");
                    supportedMetadataTypes[ConfiguredStream] = requestedMetaData.ToArray(); 
                }
                return PixelSensorFeature.StartSensor(SensorId, new[] { ConfiguredStream },
                supportedMetadataTypes);
            }

            public void Initialize()
            {
            }

            private void GetStringFromMetaData(in PixelSensorMetaData[] frameMetaData, in StringBuilder builder, out PixelSensorPinholeIntrinsics out_pinholeIntrinsics, out PixelSensorFisheyeIntrinsics out_fisheyeIntrinsics)
            {
                out_pinholeIntrinsics = null;
                out_fisheyeIntrinsics = null;
                Debug.Log("frameMetaData.Length: " + frameMetaData.Length);
                for (int i = 0; i < frameMetaData.Length; i++)
                {
                    var metaData = frameMetaData[i];
                    Debug.Log(metaData);
                    if (metaData is PixelSensorPinholeIntrinsics pinhole)
                    {
                        out_pinholeIntrinsics = pinhole;
                        Debug.Log("pinhole value assigned!");
                    }
                    if (metaData is PixelSensorFisheyeIntrinsics fisheye)
                    {
                        out_fisheyeIntrinsics = fisheye;
                        Debug.Log("fisheye value assigned!");
                    }
                    switch (metaData)
                    {
                        case PixelSensorAnalogGain analogGain:
                            builder.AppendLine($"AnalogGain: {analogGain.AnalogGain}");
                            break;
                        case PixelSensorDigitalGain digitalGain:
                            builder.AppendLine($"DigitalGain: {digitalGain.DigitalGain}");
                            break;
                        case PixelSensorExposureTime exposureTime:
                            builder.AppendLine($"ExposureTime: {exposureTime.ExposureTime:F1}");
                            break;
                        case PixelSensorDepthFrameIllumination illumination:
                            builder.AppendLine($"IlluminationType: {illumination.IlluminationType}");
                            break;
                        case PixelSensorFisheyeIntrinsics fisheyeIntrinsics:
                            {
                                builder.AppendLine($"[Fisheye Intrinsics]");
                                builder.AppendLine($"FOV: {fisheyeIntrinsics.FOV}");
                                builder.AppendLine($"Focal Length: {fisheyeIntrinsics.FocalLength}");
                                builder.AppendLine($"Principal Point: {fisheyeIntrinsics.PrincipalPoint}");
                                builder.AppendLine(
                                    $"Radial Distortion [k1, k2, k3, k4]: [{string.Join(',', fisheyeIntrinsics.RadialDistortion.Select(val => val.ToString("F3")))}]");
                                builder.AppendLine(
                                    $"Tangential Distortion [p1, p2]: [{string.Join(',', fisheyeIntrinsics.TangentialDistortion.Select(val => val.ToString("F3")))}]");
                                break;
                            }
                        case PixelSensorPinholeIntrinsics pinholeIntrinsics:
                            {
                                builder.AppendLine($"[Pinhole Intrinsics]");
                                builder.AppendLine($"FOV: {pinholeIntrinsics.FOV}");
                                builder.AppendLine($"Focal Length: {pinholeIntrinsics.FocalLength}");
                                builder.AppendLine($"Principal Point: {pinholeIntrinsics.PrincipalPoint}");
                                builder.AppendLine(
                                    $"Pinhole Distortion [k1, k2, p1, p2, k3]: [{string.Join(',', pinholeIntrinsics.Distortion.Select(val => val.ToString("F3")))}]");
                                break;
                            }
                    }
                }

            }


        }

            enum PixelSensorType
            {
                Depth,
                World,
                Eye,
                Picture
        }
    }

}
