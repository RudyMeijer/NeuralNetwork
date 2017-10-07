using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
	internal class NeuralNetwork
	{
		private int numInputs;
		private int numHidden;
		private int numOutputs;
		private List<Neuron> inputNeurons = new List<Neuron>();
		private List<Neuron> hiddenNeurons = new List<Neuron>();
		private List<Neuron> outputNeurons = new List<Neuron>();

		public NeuralNetwork(int numInputs, int numHidden, int numOutputs)
		{
			this.numInputs = numInputs;
			this.numHidden = numHidden;
			this.numOutputs = numOutputs;
			for (int i = 0; i < numInputs; i++) this.inputNeurons.Add(new Neuron());
			for (int i = 0; i < numHidden; i++) this.hiddenNeurons.Add(new Neuron());
			for (int i = 0; i < numOutputs; i++) this.outputNeurons.Add(new Neuron());
		}

		internal double[] Train(double[] trainData, int maxEpochs, double learnRate)
		{
			//throw new NotImplementedException();
			return null;
		}
	}
}