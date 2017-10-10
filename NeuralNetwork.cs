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
		internal double[] Train(double[] trainData, int maxEpochs, double learnRate)
		{
			var mse = 1.0;
			int epoch = 0;
			int idx = 0;
			while (epoch < maxEpochs && mse > 0.01 && (Inputs.Length + idx) < trainData.Length)
			{
				this.Inputs = SetInputs(trainData, idx);
				this.ExpectedOutput = trainData[Inputs.Length + idx];
				ComputeOutputs();
				//UpdateWeights();
				//mse = //GetMeansSquareError();
				epoch++; idx++;
			}

			return null;//GetWeigths();
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