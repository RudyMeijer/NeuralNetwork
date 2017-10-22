using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	class Program
	{
		static void Main(string[] args)
		{
			var numInputs = 1; // Equal to rolling window size.
			var numHidden = 2;
			var numOutputs = 1;
			var nn = new NeuralNetwork(numInputs, numHidden, numOutputs);
			//
			// Get the training data from a textfile.
			//
			double[] trainData = GetTrainData("Data\\TrainData.txt");
			//
			// Normalize the data.
			//
			Normalize(trainData);
			//
			// Train the neural network with traning data.
			//
			trainData = new double[] { 1, 1 }; // Size is numInputs + 1

			nn.Train(trainData, maxEpochs: 100, learnRate: 0.1);
			nn.ShowNeuralNetwork();
			Debug.WriteLine("================== Done ==================");
		}

		/// <summary>
		/// Normalize data by computing (x - mean) / standard deviation for each value.
		/// </summary>
		/// <param name="trainData"></param>
		private static void Normalize(double[] trainData)
		{
			var sum = 0d;
			var mean = trainData.Average();

			for (int i = 0; i < trainData.Length; i++) sum += (trainData[i] - mean) * (trainData[i] - mean);

			var StandardDeviation = Math.Sqrt(sum / trainData.Length - 1);

			for (int i = 0; i < trainData.Length; i++) trainData[i] = (trainData[i] - mean) / StandardDeviation;
		}

		private static double[] GetTrainData(string fileName)
		{
			var separator = ';';
			var values = new List<double>();
			using (var sr = new StreamReader(fileName))
			{
				while (!sr.EndOfStream)
				{
					var line = sr.ReadLine();
					if (line.Contains(separator))
					{
						var s = line.Split(separator)[1];
						Double.TryParse(s, out double result);
						values.Add(result);
					}
				}
			}
			return values.ToArray();
		}
	}
}
