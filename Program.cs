﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	public class Program
	{
		public static bool DEBUG = true;

		static void Main(string[] args)
		{
			var nn = new NeuralNetwork(numInputs: 1, numHidden: 2, numOutputs: 1);
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
			ShowNeuralNetwork(nn);
			Debug.WriteLine("================== Done ==================");
		}
		public static void ShowNeuralNetwork(NeuralNetwork nn)
		{
			if (!DEBUG) return;
			// Show input values.
			if (nn.numHidden == 0)
			{
				Debug.Write($"Inputs: {nn.outputNeurons[0].Inputs} W = {nn.outputNeurons[0].Weights} ");
				Debug.WriteLineIf(nn.numOutputs > 1, "");
			}
			else
				Debug.WriteLine($"Inputs: {nn.hiddenNeurons[0].Inputs}");
			for (int k = 0; k < nn.numOutputs; k++)
			{
				for (int j = 0; j < nn.numHidden; j++)
				{
					Debug.WriteLine($"Weights {nn.hiddenNeurons[j].Weights} output H{j,-2}: {nn.hiddenNeurons[j].ExecuteIW(),6:f2}/{nn.Hidden[j],5:f2} * {nn.outputNeurons[k].Weights[j]:f2} hGrad {nn.hGrads[j]:f2}");
				}
				Debug.WriteLine($"Output{k} = {nn.outputNeurons[k].ExecuteIW():f2}/{nn.Output[k]:f2} Target = {nn.ExpectedOutput:f2} ograd {nn.oGrads[k]:f2} learnRate {nn.LearnRate} epoch {nn.Epoch} mse = {nn.mse:f2}");
			}
			Debug.WriteIf(nn.mse >= 100, "NO CONVERGENCE ");
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
