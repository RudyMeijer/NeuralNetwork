using Lib;
using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
	internal class NeuralNetwork
	{
		private int numInputs;
		private Vector Inputs;
		private Vector Hidden;
		private Vector Output;
		private double ExpectedOutput;
		private List<Neuron> hiddenNeurons = new List<Neuron>();
		private List<Neuron> outputNeurons = new List<Neuron>();
		/// <summary>
		/// Create Neural network with Hidden- and Output neurons.
		/// </summary>
		/// <param name="numInputs"></param>
		/// <param name="numHidden"></param>
		/// <param name="numOutputs"></param>
		public NeuralNetwork(int numInputs, int numHidden, int numOutputs)
		{
			this.numInputs = numInputs;
			this.Inputs = new Vector(numInputs);
			this.Hidden = new Vector(numHidden);
			this.Output = new Vector(numOutputs);
			for (int i = 0; i < numHidden; i++) this.hiddenNeurons.Add(new Neuron(this.Inputs));
			for (int i = 0; i < numOutputs; i++) this.outputNeurons.Add(new Neuron(this.Hidden));
		}
		/// <summary>
		/// Train the neural network with train data.
		/// </summary>
		/// <param name="trainData"></param>
		/// <param name="maxEpochs"></param>
		/// <param name="learnRate"></param>
		/// <returns></returns>
		internal double[,] Train(double[] trainData, int maxEpochs, double learnRate)
		{
			var mse = 1.0;
			int epoch = 0;
			int idx = 0;
			while (epoch < maxEpochs && mse > 0.01 && (Inputs.Length + idx) < trainData.Length)
			{
				this.Inputs = SetInputs(trainData, idx);
				this.ExpectedOutput = trainData[Inputs.Length + idx];
				ComputeOutputs();
				BackPropagation(this.ExpectedOutput, learnRate);//UpdateWeights();
																//mse = //GetMeansSquareError();
				++epoch; ++idx;
			}

			return GetWeigths();
		}

		/// <summary>
		/// Update the weights and biases using back-propagation.
		/// </summary>
		private void BackPropagation(double expectedOutput, double learnRate)
		{
			//
			// 1. compute output gradients
			//
			var oGrads = new Double[Output.Length];
			var hGrads = new Double[Hidden.Length];
			for (int i = 0; i < oGrads.Length; ++i)
			{
				//For sigmoid activation, the derivative of y = log-sigmoid(x) is y * (1 - y)
				double derivative = Output[i] * (1 - Output[i]);
				// 'mean squared error version' includes (1-y)(y) derivative
				oGrads[i] = derivative * (ExpectedOutput - Output[i]);
			}

			// 2. compute hidden gradients
			for (int i = 0; i < hGrads.Length; ++i)
			{
				// derivative of tanh = (1 - y) * (1 + y)
				double derivative = Output[i] * (1 - Output[i]);
				double sum = 0.0;
				for (int j = 0; j < Output.Length; ++j) // each hidden delta is the sum of numOutput terms
				{
					double x = oGrads[j] * hiddenNeurons[i].Weights[j];// hoWeights[i][j];
					sum += x;
				}
				hGrads[i] = derivative * sum;
			}
			// 3a. update hidden weights (gradients must be computed right-to-left but weights
			// can be updated in any order)
			//for (int i = 0; i < ihWeights.Length; ++i) // 0..2 (3)
			//{
			//	for (int j = 0; j < ihWeights[0].Length; ++j) // 0..3 (4)
			//	{
			//		double delta = learnRate * hGrads[j] * inputs[i]; // compute the new delta
			//		ihWeights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
			//								  // now add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
			//		ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j];
			//		ihWeights[i][j] -= (weightDecay * ihWeights[i][j]); // weight decay
			//		ihPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum 
			//	}
			//}

			//// 3b. update hidden biases
			//for (int i = 0; i < hBiases.Length; ++i)
			//{
			//	double delta = learnRate * hGrads[i] * 1.0; // t1.0 is constant input for bias; could leave out
			//	hBiases[i] += delta;
			//	hBiases[i] += momentum * hPrevBiasesDelta[i]; // momentum
			//	hBiases[i] -= (weightDecay * hBiases[i]); // weight decay
			//	hPrevBiasesDelta[i] = delta; // don't forget to save the delta
			//}

		}

		private double[,] GetWeigths()
		{
			double[,] weights = new Double[hiddenNeurons.Count, Inputs.Length];
			for (int i = 0; i < hiddenNeurons.Count; i++)
			{
				for (int j = 0; j < Inputs.Length; j++)
				{
					weights[i, j] = hiddenNeurons[i].Weights[j];
				}
			}
			for (int i = 0; i < outputNeurons.Count; i++)
			{
				for (int j = 0; j < Hidden.Length; j++)
				{
					weights[i, j] = outputNeurons[i].Weights[j];
				}
			}
			return weights;
		}

		private void ComputeOutputs()
		{
			for (int i = 0; i < hiddenNeurons.Count; i++) this.Hidden[i] = hiddenNeurons[i].Execute();
			for (int i = 0; i < outputNeurons.Count; i++) this.Output[i] = outputNeurons[i].Execute();
		}

		private Vector SetInputs(double[] trainData, int idx)
		{
			for (int i = 0; i < Inputs.Length; i++)
			{
				Inputs[i] = trainData[idx + i];
			}
			return Inputs;
		}
	}
}