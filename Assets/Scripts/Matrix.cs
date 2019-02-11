using System;
using System.Text.RegularExpressions;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Matrix2
{

    public float[,] mat;

    public Matrix2()
    {
        mat = new float[2, 2];
    }

    public float this[int iRow, int iCol]
    {
        get { return mat[iRow, iCol]; }
        set { mat[iRow, iCol] = value; }
    }

    public static Matrix2 ZeroMatrix()
    {
        Matrix2 matrix = new Matrix2();
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                matrix[i, j] = 0;
        return matrix;
    }

    public static Matrix2 IdentityMatrix()
    {
        Matrix2 matrix = ZeroMatrix();
        for (int i = 0; i < Math.Min(2, 2); i++)
            matrix[i, i] = 1;
        return matrix;
    }

    public static Matrix2 Transpose(Matrix2 m)
    {
        Matrix2 t = new Matrix2();
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                t[j, i] = m[i, j];
        return t;
    }

    private static Matrix2 Multiply(float n, Matrix2 m)
    {
        Matrix2 r = new Matrix2();
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                r[i, j] = m[i, j] * n;
        return r;
    }
    private static Matrix2 Add(Matrix2 m1, Matrix2 m2) 
    {
        Matrix2 r = new Matrix2();
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                r[i, j] = m1[i, j] + m2[i, j];
        return r;
    }

    public void MakeRotation(float angle)
    {
        mat[0,0] = Mathf.Cos(angle);
        mat[1,0] = Mathf.Sin(angle);
        mat[0,1] = -mat[1,0];
        mat[1,1] = mat[0,0];
    }

    // Operators
    public static Matrix2 operator -(Matrix2 m)
    { return Matrix2.Multiply(-1, m); }

    public static Matrix2 operator +(Matrix2 m1, Matrix2 m2)
    { return Matrix2.Add(m1, m2); }

    public static Matrix2 operator -(Matrix2 m1, Matrix2 m2)
    { return Matrix2.Add(m1, -m2); }

    public static Matrix2 operator *(float n, Matrix2 m)
    { return Matrix2.Multiply(n, m); }

    public static Vector2 operator *(Vector2 vec, Matrix2 m)
    {
        return new Vector2(
            vec[0] * m[0, 0] + vec[1] * m[0, 1],
            vec[0] * m[1, 0] + vec[1] * m[1, 1]);
    }
}