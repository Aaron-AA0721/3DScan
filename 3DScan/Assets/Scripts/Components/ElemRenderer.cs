#nullable enable
using System.Linq;
using UnityEngine;


public class ElemRenderer : MonoBehaviour
{
    private Mesh? mesh;

    public Mesh UpdateMesh(
        Vector3[] arrVertices,
        int nPointsToRender,
        int nPointsRendered,
        Color[] pointColor
    )
    {
        var nPoints = Mathf.Min(nPointsToRender, arrVertices.Length - nPointsRendered);
        nPoints = Mathf.Min(nPoints, 65535);

        var points = new Vector3[nPoints]; //arrVertices.Skip(nPointsRendered).Take(nPoints).ToArray();
        var indices = new int[nPoints];
        var colors = new Color[nPoints]; //pointColor.Skip(nPointsRendered).Take(nPoints).ToArray();

        for (var i = 0; i < nPoints; i++)
        {
            points[i] = arrVertices[nPointsRendered + i];
            indices[i] = i;
            colors[i] = pointColor[nPointsRendered + i];
        }

        if (mesh != null)
            Destroy(mesh);
        
        // create sphere mesh for each point
        mesh = new Mesh();
        mesh.SetVertices(points);
        mesh.SetColors(colors);
        mesh.SetIndices(indices, MeshTopology.Points, 0);

        GetComponent<MeshFilter>().mesh = mesh;
        // GetComponent<MeshCollider>().sharedMesh = mesh;
        // GetComponent<MeshCollider>().isTrigger = true;
        return mesh;
    }
}
