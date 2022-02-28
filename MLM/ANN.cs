using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace MLM
{
    [JsonObject(MemberSerialization.OptIn)]
    public class ANN
    {
        [JsonProperty]
        private string Name { get; set; }
        [JsonProperty]
        List<Matrix> W = new List<Matrix>();
        [JsonProperty]
        List<Matrix> B = new List<Matrix>();
        List<Matrix> FPZ, FPA;

        private ANN() { }

        public ANN(int input, int[] innerLayers, int output, string name = "")
        {
            if (innerLayers.Length <= 0)
            {
                throw new Exception("There must be at least 1 inner layer of 'n' units");
            }
            Name = name;
            int i = 0;
            W.Add(new Matrix(input, innerLayers[i], true));
            B.Add(new Matrix(1, innerLayers[i++], true));
            for (; i < innerLayers.Length; i++)
            {
                W.Add(new Matrix(innerLayers[i - 1], innerLayers[i], true));
                B.Add(new Matrix(1, innerLayers[i], true));
            }
            W.Add(new Matrix(innerLayers[--i], output, true));
            B.Add(new Matrix(1, output, true));
        }

        private static Matrix Sigmoid(Matrix A)
        {
            return 1 / (1 + (-A).Exp());
        }

        private static Matrix DSigmoid(Matrix A)
        {
            Matrix sigma_z = Sigmoid(A);
            Matrix g = sigma_z * (1 - sigma_z);
            return g;
        }

        public void FP(Matrix xi)
        {
            if (xi.Width != W[0].Height)
                throw new Exception("Input width does not match ANN input height.");
            FPZ = new List<Matrix>();
            FPA = new List<Matrix>();
            Matrix A = xi;
            FPA.Add(xi);
            for (int i = 0; i < W.Count; i++)
            {
                Matrix Zi = A & W[i];
                if (i < W.Count - 1)
                    FPZ.Add(Zi);
                A = Sigmoid(Zi + B[i]);
                FPA.Add(A);
            }
        }

        public List<Matrix>[] BP(Matrix Y)
        {
            int k = W.Count - 1;
            List<Matrix> D = new List<Matrix>();
            Matrix T = FPA[FPA.Count - 1] - Y;
            D.Add(T);
            for (int j = FPZ.Count - 1; j >= 0; j--)
            {
                T = (T & (W[k--].T)) * DSigmoid(FPZ[j]);
                D.Add(T);
            }
            List<Matrix> dW = new List<Matrix>();
            List<Matrix> dB = new List<Matrix>();
            for (int i = 0; i < W.Count; i++)
            {
                dW.Add(FPA[i].T & D[W.Count - 1 - i]);
                dB.Add(B[i] = D[W.Count - 1 - i]);
            }
            return new List<Matrix>[] { dW, dB };
        }

        public void Train(List<Matrix> X, List<Matrix> Y, int iter, double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 10e-8)
        {
            /*
             * This optimizer is based of 'diffGrad' a variant of adam optimizer.
             * https://ieeexplore.ieee.org/document/8939562
             */
            List<Matrix> m_dW = new List<Matrix>();
            List<Matrix> m_dB = new List<Matrix>();
            List<Matrix> v_dW = new List<Matrix>();
            List<Matrix> v_dB = new List<Matrix>();
            foreach (Matrix dWi in W)
            {
                m_dW.Add(new Matrix(dWi.Height, dWi.Width));
                v_dW.Add(new Matrix(dWi.Height, dWi.Width));
            }
            foreach (Matrix dBi in B)
            {
                m_dB.Add(new Matrix(dBi.Height, dBi.Width));
                v_dB.Add(new Matrix(dBi.Height, dBi.Width));
            }
            for (int i = 0; i < iter; i++)
            {
                FP(X[i % X.Count]);
                List<Matrix>[] deriv = BP(Y[i % Y.Count]);
                List<Matrix> dW = deriv[0];
                List<Matrix> dB = deriv[1];

                for (int j = 0; j < m_dW.Count; j++)
                {
                    m_dW[j] = beta1 * m_dW[j] + ((1 - beta1) * dW[j]);
                    m_dB[j] = beta1 * m_dB[j] + ((1 - beta1) * dB[j]);
                }

                for (int j = 0; j < v_dW.Count; j++)
                {
                    v_dW[j] = beta2 * v_dW[j] + ((1 - beta2) * (dW[j] * dW[j]));
                    v_dB[j] = beta2 * v_dB[j] + ((1 - beta2) * (dB[j] * dB[j]));
                }

                List<Matrix> m_dW_c = new List<Matrix>();
                for (int j = 0; j < m_dW.Count; j++)
                    m_dW_c.Add(m_dW[j] / (1 - Math.Pow(beta1, i + 1)));
                List<Matrix> m_dB_c = new List<Matrix>();
                for (int j = 0; j < m_dB.Count; j++)
                    m_dB_c.Add(m_dB[j] / (1 - Math.Pow(beta1, i + 1)));

                List<Matrix> v_dW_c = new List<Matrix>();
                for (int j = 0; j < v_dW.Count; j++)
                    v_dW_c.Add(v_dW[j] / (1 - Math.Pow(beta2, i + 1)));
                List<Matrix> v_dB_c = new List<Matrix>();
                for (int j = 0; j < v_dW.Count; j++)
                    v_dB_c.Add(v_dB[j] / (1 - Math.Pow(beta2, i + 1)));

                List<Matrix> xi_dW = new List<Matrix>();
                for (int j = 0; j < W.Count; j++)
                    xi_dW.Add(Sigmoid((W[j] - dW[j]).Abs()));
                List<Matrix> xi_dB = new List<Matrix>();
                for (int j = 0; j < B.Count; j++)
                    xi_dB.Add(Sigmoid((B[j] - dB[j]).Abs()));

                for (int j = 0; j < W.Count; j++)
                    W[j] = W[j] - (alpha * xi_dW[j] * m_dW_c[j]) / (v_dW_c[j].Sqrt() + epsilon);
                for (int j = 0; j < B.Count; j++)
                    B[j] = B[j] - (alpha * xi_dB[j] * m_dB_c[j]) / (v_dB_c[j].Sqrt() + epsilon);
            }
        }

        public Matrix Infer(Matrix Xi)
        {
            Matrix H = Xi;
            for (int i = 0; i < W.Count; i++)
                H = Sigmoid((H & W[i]) + B[i]);
            return H;
        }

        public override string ToString()
        {
            string rep = $"{Name} - [{W[0].Height}";
            string content = "";
            foreach (Matrix Wi in W)
            {
                rep += $",{Wi.Width}";
                content += $"{Wi}\n";
            }
            rep += "]\n\n" + content;
            return rep;
        }

        public void Save(string fileName)
        {
            string jsonString = JsonConvert.SerializeObject(this, Formatting.Indented);
            File.WriteAllText(fileName, jsonString);
        }

        public static ANN Load(string fileName)
        {
            string jsonString = File.ReadAllText(fileName);
            JObject result = JObject.Parse(jsonString);
            List<Matrix> W = new List<Matrix>();
            IList<JToken> resultW = result["W"].Children().ToList();
            LoadDeserializedListMatrix(W, resultW);
            List<Matrix> B = new List<Matrix>();
            IList<JToken> resultB = result["B"].Children().ToList();
            LoadDeserializedListMatrix(B, resultB);
            string name = result["Name"].Value<string>();
            ANN ann = new ANN();
            ann.Name = name;
            ann.W = W;
            ann.B = B;
            return ann;
        }

        private static void LoadDeserializedListMatrix(List<Matrix> M, IList<JToken> resultM)
        {
            foreach (JToken result in resultM)
            {
                Matrix xi = new Matrix(result["M"].Value<JArray>().ToObject<double[,]>(), true);
                M.Add(xi);
            }
        }
    }
}
