using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Newtonsoft.Json;

namespace MLM
{
    [JsonObject(MemberSerialization.OptIn)]
    public class Matrix
    {
        Random rand = new Random();
        [JsonProperty]
        private double[,] M;
        private bool isIdentity = false;
        public int Height { get; }
        public int Width { get; }
        public int Length { get; }

        public Matrix(double[,] Matrix, bool effecient = false)
        {
            Height = Matrix.GetLength(0);
            Width = Matrix.GetLength(1);
            Length = Matrix.Length;
            if (effecient)
            {
                M = Matrix;
                return;
            }
            M = new double[Height, Width];
            for (int x = 0; x < Height; x++)
                for (int y = 0; y < Width; y++)
                    M[x, y] = Matrix[x, y];
        }

        public Matrix(int height, int width, bool randomInit = false, double initialValue = 0)
        {
            Height = height;
            Width = width;
            M = new double[height, width];
            if (initialValue != 0)
                for (int x = 0; x < Height; x++)
                    for (int y = 0; y < Width; y++)
                        M[x, y] = initialValue;
            Length = M.Length;
            if (!randomInit) return;
            for (int x = 0; x < Height; x++)
                for (int y = 0; y < Width; y++)
                    M[x, y] = rand.NextDouble();
        }

        public Matrix(int N, bool identity = false, bool randomInit = false, double initialValue = 0)
        {
            Width = Height = N;
            Length = M.Length;
            M = new double[N, N];
            if (initialValue != 0)
                for (int x = 0; x < Height; x++)
                    for (int y = 0; y < Width; y++)
                        M[x, y] = initialValue;
            if (!identity) return;
            for (int x = 0; x < N; x++)
                M[x, x] = 1;
            if (!randomInit) return;
            for (int x = 0; x < Height; x++)
                for (int y = 0; y < Width; y++)
                    M[x, y] = rand.NextDouble();
        }

        private static int[] checkRowCol(Matrix A, Matrix B)
        {
            if (A.Height != B.Height && A.Width != B.Width)
            {
                throw new Exception("Matrix dimension does not match.");
            }
            return new int[2] { A.Height, A.Width };
        }

        public Matrix Exp()
        {
            Matrix C = new Matrix(Height, Width);
            for (int x = 0; x < Height; x++)
                for (int y = 0; y < Width; y++)
                    C[x, y] = Math.Exp(M[x, y]);
            return C;
        }

        public double Mean()
        {
            double sum = 0;
            for (int x = 0; x < Height; x++)
                for (int y = 0; y < Width; y++)
                    sum += M[x, y];
            return sum / M.Length;
        }

        public Matrix Abs()
        {
            Matrix C = new Matrix(Height, Width);
            for (int x = 0; x < Height; x++)
                for (int y = 0; y < Width; y++)
                    C[x, y] = Math.Abs(M[x, y]);
            return C;
        }

        public Matrix Sqrt()
        {
            Matrix C = new Matrix(Height, Width);
            for (int x = 0; x < Height; x++)
                for (int y = 0; y < Width; y++)
                    C[x, y] = Math.Sqrt(M[x, y]);
            return C;
        }

        public static Matrix operator -(Matrix A)
        {
            Matrix C = new Matrix(A.Height, A.Width);
            for (int x = 0; x < A.Height; x++)
                for (int y = 0; y < A.Width; y++)
                    C[x, y] = -A[x, y];
            return C;
        }

        public static Matrix operator +(Matrix A, Matrix B)
        {
            int[] dimension = checkRowCol(A, B);
            Matrix C = new Matrix(dimension[0], dimension[1]);
            for (int x = 0; x < dimension[0]; x++)
                for (int y = 0; y < dimension[1]; y++)
                    C[x, y] = A[x, y] + B[x, y];
            return C;
        }

        public static Matrix operator +(double a, Matrix B)
        {
            Matrix C = new Matrix(B.Height, B.Width);
            for (int x = 0; x < B.Height; x++)
                for (int y = 0; y < B.Width; y++)
                    C[x, y] = a + B[x, y];
            return C;
        }

        public static Matrix operator +(Matrix A, double b)
        {
            Matrix C = new Matrix(A.Height, A.Width);
            for (int x = 0; x < A.Height; x++)
                for (int y = 0; y < A.Width; y++)
                    C[x, y] = A[x, y] + b;
            return C;
        }

        public static Matrix operator -(Matrix A, Matrix B)
        {
            int[] dimension = checkRowCol(A, B);
            Matrix C = new Matrix(dimension[0], dimension[1]);
            for (int x = 0; x < dimension[0]; x++)
                for (int y = 0; y < dimension[1]; y++)
                    C[x, y] = A[x, y] - B[x, y];
            return C;
        }

        public static Matrix operator -(double a, Matrix B)
        {
            Matrix C = new Matrix(B.Height, B.Width);
            for (int x = 0; x < B.Height; x++)
                for (int y = 0; y < B.Width; y++)
                    C[x, y] = a - B[x, y];
            return C;
        }

        public static Matrix operator -(Matrix A, double b)
        {
            Matrix C = new Matrix(A.Height, A.Width);
            for (int x = 0; x < A.Height; x++)
                for (int y = 0; y < A.Width; y++)
                    C[x, y] = A[x, y] - b;
            return C;
        }

        public static Matrix operator *(Matrix A, Matrix B)
        {
            /*
             * Scalar product
             */
            int[] dimension = checkRowCol(A, B);
            Matrix C = new Matrix(dimension[0], dimension[1]);
            for (int x = 0; x < dimension[0]; x++)
                for (int y = 0; y < dimension[1]; y++)
                    C[x, y] = A[x, y] * B[x, y];
            return C;
        }

        public static Matrix operator *(double a, Matrix B)
        {
            /*
             * Scalar product
             */
            Matrix C = new Matrix(B.Height, B.Width);
            for (int x = 0; x < B.Height; x++)
                for (int y = 0; y < B.Width; y++)
                    C[x, y] = a * B[x, y];
            return C;
        }

        public static Matrix operator *(Matrix A, double b)
        {
            /*
             * Scalar product
             */
            Matrix C = new Matrix(A.Height, A.Width);
            for (int x = 0; x < A.Height; x++)
                for (int y = 0; y < A.Width; y++)
                    C[x, y] = A[x, y] * b;
            return C;
        }

        public static Matrix operator /(Matrix A, Matrix B)
        {
            int[] dimension = checkRowCol(A, B);
            Matrix C = new Matrix(dimension[0], dimension[1]);
            for (int x = 0; x < dimension[0]; x++)
                for (int y = 0; y < dimension[1]; y++)
                {
                    if (B[x, y] == 0)
                    {
                        throw new DivideByZeroException($"[{x},{y}]:{A[x, y]}/{B[x, y]}");
                    }
                    C[x, y] = A[x, y] / B[x, y];
                }
            return C;
        }

        public static Matrix operator /(double a, Matrix B)
        {
            Matrix C = new Matrix(B.Height, B.Width);
            for (int x = 0; x < B.Height; x++)
                for (int y = 0; y < B.Width; y++)
                {
                    if (B[x, y] == 0)
                    {
                        throw new DivideByZeroException($"[{x},{y}]:{a}/{B[x, y]}");
                    }
                    C[x, y] = a / B[x, y];
                }
            return C;
        }

        public static Matrix operator /(Matrix A, double b)
        {
            Matrix C = new Matrix(A.Height, A.Width);
            for (int x = 0; x < A.Height; x++)
                for (int y = 0; y < A.Width; y++)
                {
                    if (b == 0)
                    {
                        throw new DivideByZeroException($"[{x},{y}]:{A[x, y]}/{b}");
                    }
                    C[x, y] = A[x, y] / b;
                }
            return C;
        }

        public static Matrix operator &(Matrix A, Matrix B)
        {
            /*
             * Dot product
             */
            if (A.Width != B.Height && A.M.Length != 1 && B.M.Length != 1)
            {
                throw new Exception("Matrix dimension does not match.");
            }
            if (A.isIdentity)
            {
                Matrix I = new Matrix(B.Height, B.Width);
                for (int x = 0; x < B.Height; x++)
                    for (int y = 0; y < B.Width; y++)
                        I[x, y] = B[x, y];
                return I;
            }
            if (B.isIdentity)
            {
                Matrix I = new Matrix(A.Height, A.Width);
                for (int x = 0; x < A.Height; x++)
                    for (int y = 0; y < A.Width; y++)
                        I[x, y] = A[x, y];
                return I;
            }
            if (A.M.Length == 1)
            {
                Matrix I = new Matrix(B.Height, B.Width);
                for (int x = 0; x < B.Height; x++)
                    for (int y = 0; y < B.Width; y++)
                        I[x, y] = B[x, y] * A[0, 0];
                return I;
            }
            if (B.M.Length == 1)
            {
                Matrix I = new Matrix(A.Height, A.Width);
                for (int x = 0; x < A.Height; x++)
                    for (int y = 0; y < A.Width; y++)
                        I[x, y] = A[x, y] * B[0, 0];
                return I;
            }
            Matrix C = new Matrix(A.Height, B.Width);
            for (int x = 0; x < C.Height; x++)
                for (int y = 0; y < C.Width; y++)
                {
                    double temp = 0;
                    for (int z = 0; z < A.Width; z++)
                        temp += A[x, z] * B[z, y];
                    C[x, y] = temp;
                }
            return C;
        }

        public static Matrix operator &(Matrix A, double b)
        {
            /*
             * Dot product
             */
            Matrix C = new Matrix(A.Height, A.Width);
            for (int x = 0; x < A.Height; x++)
                for (int y = 0; y < A.Width; y++)
                    C[x, y] = A[x, y] * b;
            return C;
        }

        public static Matrix operator &(double a, Matrix B)
        {
            /*
             * Dot product
             */
            Matrix C = new Matrix(B.Height, B.Width);
            for (int x = 0; x < B.Height; x++)
                for (int y = 0; y < B.Width; y++)
                    C[x, y] = B[x, y] * a;
            return C;
        }

        public override string ToString()
        {
            int row = Height;
            int col = Width;
            string rep = "[" + row + "," + col + "]\n";
            for (int x = 0; x < row; x++)
            {
                for (int y = 0; y < col; y++)
                    rep += M[x, y] + (y < col - 1 ? "," : "\n");
            }
            return rep;
        }

        public double this[int row, int column]
        {
            get => M[row, column];
            set => M[row, column] = value;
        }

        public Matrix T
        {
            get
            {
                int row = Height;
                int column = Width;

                Matrix transposed = new Matrix(column, row);

                for (int x = 0; x < row; x++)
                    for (int y = 0; y < column; y++)
                        transposed[y, x] = M[x, y];

                return transposed;
            }
        }
    }
}
