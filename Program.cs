using System;
using System.Collections.Generic;
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
			var numInputs = 4; // Equal to rolling window size.
			var numOutputs = 1;
			var numHidden = 12;
			var nn = new NeuralNetwork(numInputs, numHidden, numOutputs);

			double[] trainData = GetTrainData("Data\\TrainData.txt"); 

			var maxEpochs = 10000;
			var learnRate = 0.01;
			double[] weights = nn.Train(trainData, maxEpochs, learnRate);
			ShowVector(weights, 2, 10, true);
		}

		private static void ShowVector(double[] weights, int decimals, int rowLength, bool v3)
		{
			//throw new NotImplementedException();
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
