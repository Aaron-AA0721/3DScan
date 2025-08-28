using g4;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using UnityEngine;

namespace Components
{
    public class PointCloudRenderer : MonoBehaviour
    {
        public int maxSize = 65535;
        public float pointSize = 0.04f;
        public GameObject pointCloudElem = null!;
        public Material pointCloudMaterial = null!;
        
        private List<GameObject> elems = null!;
        private List<(int beg, int end)> elemIndices = new List<(int, int)>();
        private static readonly int PointSize = Shader.PropertyToID("_PointSize");

        void Start()
        {
            elems = new List<GameObject>();
            UpdatePointSize();
        }

        void Update()
        {
            if (transform.hasChanged)
            {
                UpdatePointSize();
                transform.hasChanged = false;
            }
        }

        public void UpdatePointSize()
        {
            pointCloudMaterial.SetFloat(PointSize, pointSize * transform.localScale.x);
        }

        public void Render(Vector3[] arrVertices, Color[] pointColor)
        {
            int nPoints,
                nChunks;
            if (arrVertices == null)
            {
                nPoints = 0;
                nChunks = 0;
            }
            else
            {
                nPoints = arrVertices.Length;
                nChunks = 1 + nPoints / maxSize;
            }

            if (elems.Count < nChunks)
                AddElems(nChunks - elems.Count);
            if (elems.Count > nChunks)
                RemoveElems(elems.Count - nChunks);

            int offset = 0;
            for (var i = 0; i < nChunks; i++)
            {
                int nPointsToRender = Mathf.Min(maxSize, nPoints - offset);

                ElemRenderer elemRenderer = elems[i].GetComponent<ElemRenderer>();
                Mesh mesh = elemRenderer.UpdateMesh(arrVertices, nPointsToRender, offset, pointColor);
                
                // collider for raycast-based point picking (testing)
                //var elemCollider = elems[i].GetComponent<MeshCollider>();
                //elemCollider.sharedMesh = mesh;
                
                elems[i].layer = gameObject.layer;
                offset += nPointsToRender;
            }
        }
        public void AppendRender(Vector3[] arrVertices, Color[] pointColor)
        {
            int nPoints,
                nChunks;
            if (arrVertices == null)
            {
                nPoints = 0;
                nChunks = 0;
            }
            else
            {
                nPoints = arrVertices.Length;
                nChunks = 1 + nPoints / maxSize;
            }
            AddElems(nChunks);
            // if (elems.Count < nChunks)
            //     AddElems(nChunks - elems.Count);
            // if (elems.Count > nChunks)
            //     RemoveElems(elems.Count - nChunks);

            int offset = 0;
            int beg = elemIndices.Last().beg;
            int end = elemIndices.Last().end;
            for (var i = beg; i < end; i++)
            {
                int nPointsToRender = Mathf.Min(maxSize, nPoints - offset);

                ElemRenderer elemRenderer = elems[i].GetComponent<ElemRenderer>();
                Mesh mesh = elemRenderer.UpdateMesh(arrVertices, nPointsToRender, offset, pointColor);

                // collider for raycast-based point picking (testing)
                //var elemCollider = elems[i].GetComponent<MeshCollider>();
                //elemCollider.sharedMesh = mesh;

                elems[i].layer = gameObject.layer;
                offset += nPointsToRender;
            }
        }
        void AddElems(int nElems)
        {
            elemIndices.Add((elems.Count, elems.Count + nElems));
            for (int i = 0; i < nElems; i++)
            {
                GameObject newElem = GameObject.Instantiate(pointCloudElem, transform, true);
                newElem.transform.localPosition = new Vector3(0.0f, 0.0f, 0.0f);
                newElem.transform.localRotation = Quaternion.identity;
                newElem.transform.localScale = new Vector3(1.0f, 1.0f, 1.0f);

                elems.Add(newElem);
            }
        }

        void RemoveElems(int nElems)
        {
            for (int i = 0; i < nElems; i++)
            {
                Destroy(elems[0]);
                elems.Remove(elems[0]);
            }
        }

        public void Clear()
        {
            var elements = elems.ToList();
            elems.Clear();
            foreach (GameObject elem in elements)
            {
                Destroy(elem);
            }
        }
    }
}
