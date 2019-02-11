using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class ProceduralCone
{
    public int nstrip = 16;
    public float height = 1.0f;

    private Mesh mesh;
    private Vector3 position = Vector3.zero;

    private List<Vector3> vertices = new List<Vector3>();
    private List<int> triangles = new List<int>();
    private List<Vector2> uvs = new List<Vector2>();
    private List<Vector3> normals = new List<Vector3>();

    public Mesh Mesh
    {
        get { return mesh; }
    }

    public ProceduralCone(Vector3 position)
    {
        mesh = new Mesh();
        this.position = position;

        vertices.AddRange(GetVertices());
        triangles.AddRange(GetTriangles());

        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateNormals();
    }

    private Vector3[] GetVertices()
    {
        float two_pi = 2 * 3.14159f;
        float delta = two_pi / nstrip;

        Vector3[] vertices = new Vector3[nstrip + 1];


        float angle = 0;
        for (int i = 0; i < nstrip; ++i, angle += delta)
        {
            vertices[i] = new Vector3(Mathf.Sin(angle), 0.0f, Mathf.Cos(angle));
        }
        vertices[nstrip] = new Vector3(0.0f, height, 0.0f);

        return vertices;
    }

    private int[] GetTriangles()
    {
        int[] triangles = new int[(nstrip * 3)];

        int j = 0;
        int k = 1;
        for (int i = 0; i < nstrip - 1; i++, j += 3, k++)
        {
            triangles[j] = k;
            triangles[j + 1] = nstrip;
            triangles[j + 2] = i;
        }
        triangles[j] = 0;
        triangles[j + 1] = nstrip;
        triangles[j + 2] = nstrip - 1;

        return triangles;
    }
}
