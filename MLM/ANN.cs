using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace MLM
{
    [JsonObject(MemberSerialization.OptIn)]
    public class ANN
    {
        [JsonProperty]
        private string Name { get; set; }
        [JsonProperty]
        List<double[,]> Weight = new List<double[,]>();
        List<Matrix<double>> W = new List<Matrix<double>>();
        [JsonProperty]
        List<double[,]> Bias = new List<double[,]>();
        List<Matrix<double>> B = new List<Matrix<double>>();
        List<Matrix<double>> FPZ, FPA;

        private ANN() { }

        public ANN(int input, int[] innerLayers, int output, string name = "")
        {
            if (innerLayers.Length <= 0)
                throw new Exception("There must be at least 1 inner layer of 'n' units");
            Name = name;
            int i = 0;
            W.Add(Matrix<double>.Build.Random(input, innerLayers[i]));
            B.Add(Matrix<double>.Build.Random(1, innerLayers[i++]));
            for (; i < innerLayers.Length; i++)
            {
                W.Add(Matrix<double>.Build.Random(innerLayers[i - 1], innerLayers[i]));
                B.Add(Matrix<double>.Build.Random(1, innerLayers[i]));
            }
            W.Add(Matrix<double>.Build.Random(innerLayers[--i], output));
            B.Add(Matrix<double>.Build.Random(1, output));
        }

        private static Matrix<double> Sigmoid(Matrix<double> A)
        {
            return 1 / (1 + A.Negate().PointwiseExp());
        }

        private static Matrix<double> DSigmoid(Matrix<double> A)
        {
            Matrix<double> sigma_z = Sigmoid(A);
            Matrix<double> g = sigma_z.PointwiseMultiply(1 - sigma_z);
            return g;
        }

        public void FP(Matrix<double> xi)
        {
            if (xi.ColumnCount != W[0].RowCount)
                throw new Exception("Input width does not match ANN input height.");
            FPZ = new List<Matrix<double>>();
            FPA = new List<Matrix<double>>();
            Matrix<double> A = xi;
            FPA.Add(xi);
            for (int i = 0; i < W.Count; i++)
            {
                Matrix<double> Zi = A * W[i];
                if (i < W.Count - 1)
                    FPZ.Add(Zi);
                A = Sigmoid(Zi + B[i]);
                FPA.Add(A);
            }
        }

        public List<Matrix<double>>[] BP(Matrix<double> Y)
        {
            int k = W.Count - 1;
            List<Matrix<double>> D = new List<Matrix<double>>();
            Matrix<double> T = FPA[FPA.Count - 1] - Y;
            D.Add(T);
            for (int j = FPZ.Count - 1; j >= 0; j--)
            {
                T = (T * W[k--].Transpose()).PointwiseMultiply(DSigmoid(FPZ[j]));
                D.Add(T);
            }
            List<Matrix<double>> dW = new List<Matrix<double>>();
            List<Matrix<double>> dB = new List<Matrix<double>>();
            for (int i = 0; i < W.Count; i++)
            {
                dW.Add(FPA[i].Transpose() * D[W.Count - 1 - i]);
                dB.Add(D[W.Count - 1 - i]);
            }
            return new List<Matrix<double>>[] { dW, dB };
        }

        public void Train(List<Matrix<double>> X, List<Matrix<double>> Y, int iter, double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 10e-8)
        {
            /*
             * This optimizer is based of 'diffGrad' a variant of adam optimizer.
             * https://ieeexplore.ieee.org/document/8939562
             */
            List<Matrix<double>> m_dW = new List<Matrix<double>>();
            List<Matrix<double>> m_dB = new List<Matrix<double>>();
            List<Matrix<double>> v_dW = new List<Matrix<double>>();
            List<Matrix<double>> v_dB = new List<Matrix<double>>();
            foreach (Matrix<double> dWi in W)
            {
                m_dW.Add(Matrix<double>.Build.Dense(dWi.RowCount, dWi.ColumnCount));
                v_dW.Add(Matrix<double>.Build.Dense(dWi.RowCount, dWi.ColumnCount));
            }
            foreach (Matrix<double> dBi in B)
            {
                m_dB.Add(Matrix<double>.Build.Dense(dBi.RowCount, dBi.ColumnCount));
                v_dB.Add(Matrix<double>.Build.Dense(dBi.RowCount, dBi.ColumnCount));
            }
            for (int i = 0; i < iter; i++)
            {
                FP(X[i % X.Count]);
                List<Matrix<double>>[] deriv = BP(Y[i % Y.Count]);
                List<Matrix<double>> dW = deriv[0];
                List<Matrix<double>> dB = deriv[1];

                for (int j = 0; j < m_dW.Count; j++)
                {
                    m_dW[j] = beta1 * m_dW[j] + ((1 - beta1) * dW[j]);
                    m_dB[j] = beta1 * m_dB[j] + ((1 - beta1) * dB[j]);
                }

                for (int j = 0; j < v_dW.Count; j++)
                {
                    v_dW[j] = beta2 * v_dW[j] + ((1 - beta2) * dW[j].PointwisePower(2));
                    v_dB[j] = beta2 * v_dB[j] + ((1 - beta2) * dB[j].PointwisePower(2));
                }

                List<Matrix<double>> m_dW_c = new List<Matrix<double>>();
                List<Matrix<double>> m_dB_c = new List<Matrix<double>>();
                for (int j = 0; j < m_dW.Count; j++)
                {
                    m_dW_c.Add(m_dW[j] / (1 - Math.Pow(beta1, i + 1)));
                    m_dB_c.Add(m_dB[j] / (1 - Math.Pow(beta1, i + 1)));
                }

                List<Matrix<double>> v_dW_c = new List<Matrix<double>>();
                List<Matrix<double>> v_dB_c = new List<Matrix<double>>();
                for (int j = 0; j < v_dW.Count; j++)
                {
                    v_dW_c.Add(v_dW[j] / (1 - Math.Pow(beta2, i + 1)));
                    v_dB_c.Add(v_dB[j] / (1 - Math.Pow(beta2, i + 1)));
                }

                List<Matrix<double>> xi_dW = new List<Matrix<double>>();
                List<Matrix<double>> xi_dB = new List<Matrix<double>>();
                for (int j = 0; j < W.Count; j++)
                {
                    xi_dW.Add(Sigmoid((W[j] - dW[j]).PointwiseAbs()));
                    xi_dB.Add(Sigmoid((B[j] - dB[j]).PointwiseAbs()));
                }

                for (int j = 0; j < W.Count; j++)
                    W[j] = W[j] - (alpha * xi_dW[j].PointwiseMultiply(m_dW_c[j])).PointwiseDivide(v_dW_c[j].PointwiseSqrt() + epsilon);
                for (int j = 0; j < B.Count; j++)
                    B[j] = B[j] - (alpha * xi_dB[j].PointwiseMultiply(m_dB_c[j])).PointwiseDivide(v_dB_c[j].PointwiseSqrt() + epsilon);
            }
        }

        public Matrix<double> Infer(Matrix<double> Xi)
        {
            Matrix<double> H = Xi;
            for (int i = 0; i < W.Count; i++)
                H = Sigmoid(H * W[i] + B[i]);
            return H;
        }

        public override string ToString()
        {
            string rep = $"{Name} - [{W[0].RowCount}";
            string content = "";
            foreach (Matrix<double> Wi in W)
            {
                rep += $",{Wi.ColumnCount}";
                content += $"{Wi}\n";
            }
            rep += "]\n\n" + content;
            return rep;
        }

        public void Save(string fileName)
        {
            foreach (Matrix<double> xi in W)
                Weight.Add(xi.ToArray());
            foreach (Matrix<double> xi in B)
                Bias.Add(xi.ToArray());
            string jsonString = JsonConvert.SerializeObject(this, Formatting.Indented);
            File.WriteAllText(fileName, jsonString);
        }

        public static ANN Load(string fileName)
        {
            string jsonString = File.ReadAllText(fileName);
            JObject result = JObject.Parse(jsonString);
            List<Matrix<double>> W = new List<Matrix<double>>();
            IList<JToken> resultW = result["Weight"].Children().ToList();
            LoadDeserializedListMatrix(W, resultW);
            List<Matrix<double>> B = new List<Matrix<double>>();
            IList<JToken> resultB = result["Bias"].Children().ToList();
            LoadDeserializedListMatrix(B, resultB);
            string name = result["Name"].Value<string>();
            ANN ann = new ANN();
            ann.Name = name;
            ann.W = W;
            ann.B = B;
            return ann;
        }

        private static void LoadDeserializedListMatrix(List<Matrix<double>> M, IList<JToken> resultM)
        {
            foreach (JToken result in resultM)
            {
                Matrix<double> xi = Matrix<double>.Build.DenseOfArray(result.ToObject<double[,]>());
                M.Add(xi);
            }
        }
    }
}
