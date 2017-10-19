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
			var numHidden = 12;
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
			var maxEpochs = 10000;
			var learnRate = 0.01;
			nn.Train(trainData, maxEpochs, learnRate);
			//ShowNeuralNetwork(nn);
		}

		public static void ShowNeuralNetwork(NeuralNetwork nn)
		{
			// Show input values.
			Console.WriteLine($"Inputs: {nn.hiddenNeurons[0].Inputs}");
			for (int k = 0; k < nn.numOutputs; k++)
			{
				for (int j = 0; j < nn.numHidden; j++)
					Console.WriteLine($"Weights {nn.hiddenNeurons[j].Weights} output H{j,-2}: {nn.Hidden[j],6:f2} * {nn.outputNeurons[k].Weights[j]:f2}");
				//Console.WriteLine($"Output weights {nn.outputNeurons[k].Weights}");
				Console.WriteLine($"Output = {nn.Output[k]:f2} Target = {nn.ExpectedOutput:f2}");
			}
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

		//public static void ShowVector(double[,] weights)
		//{
		//	var numInputs = weights.GetUpperBound(1) + 1;
		//	var numHidden = weights.GetUpperBound(0) + 1;
		//	for (int i = 0; i < numInputs; i++)
		//	{
		//		Console.Write($"input {i}: ");
		//		for (int j = 0; j < numHidden; j++)
		//		{
		//			Console.Write(weights[j, i].ToString("N2") + " ");
		//		}
		//		Console.WriteLine();
		//	}
		//	Console.Write("output: ");
		//	for (int j = 0; j < numHidden; j++)
		//	{
		//		Console.Write($" {hi}");
		//	}
		//	Console.WriteLine("==============");
		//}

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
