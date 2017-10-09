using Lib;
using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
	internal class NeuralNetwork
	{
		private int numInputs;
		//private int numHidden;
		//private int numOutputs;
		//private List<Neuron> inputNeurons = new List<Neuron>();
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
			//this.numHidden = numHidden;
			//this.numOutputs = numOutputs;
			//for (int i = 0; i < numInputs; i++) this.inputNeurons.Add(new Neuron());
			for (int i = 0; i < numHidden; i++) this.hiddenNeurons.Add(new Neuron(numInputs));
			for (int i = 0; i < numOutputs; i++) this.outputNeurons.Add(new Neuron(numHidden));
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
			InitWeigths();
			var mse = 1.0;
			int epoch = 0;
			int idx = 0;
			while (epoch < maxEpochs && mse > 0.01)
			{
				//SetInputs(trainData, idx);
				//ComputeOutputs();
				//UpdateWeights();
				mse = //GetMeansSquareError();
				epoch++;idx++;
			}

			return null;//GetWeigths();
		}

		private void InitWeigths()
		{
			var random = new Random();
			foreach (var neuron in hiddenNeurons)
			{
				var vector = new Vector(hiddenNeurons.Count);
				for (int i = 0; i < hiddenNeurons.Count; i++)
					vector[i] = random.NextDouble();
				neuron.Weights = vector;
			}
		}
	}
}