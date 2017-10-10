﻿using System;
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
			var maxEpochs = 10000;
			var learnRate = 0.01;
			double[] weights = nn.Train(trainData, maxEpochs, learnRate);
			ShowVector(weights, 2, 10, true);
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