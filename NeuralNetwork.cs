using System;

namespace Neural.Network
{
	internal class NeuralNetwork
	{
		private int numInputs;
		private int numHidden;
		private int numOutputs;
		private Neuron inputNeurons;

		public NeuralNetwork(int numInputs, int numHidden, int numOutputs)
		{
			this.numInputs = numInputs;
			this.numHidden = numHidden;
			this.numOutputs = numOutputs;
			this.inputNeurons = new Neuron();
		}

		internal double[] Train(double[] trainData, int maxEpochs, double learnRate)
		{
			throw new NotImplementedException();
		}
	}
}