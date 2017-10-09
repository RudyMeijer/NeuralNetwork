using System;
using Lib;

namespace NeuralNetwork
{
	/// <summary>
	/// Description of Neuron.
	/// </summary>
	public class Neuron
	{
		public Vector Inputs, Weights;
		public Double Bias;
		public Neuron(int numInputs)
		{
			var random = new Random();
			this.Inputs = new Vector(numInputs);
			this.Weights = new Vector(numInputs);
			this.Bias = random.NextDouble();
			for (int i = 0; i < numInputs; i++) Weights[i] = random.NextDouble();
		}

		public double Execute()
		{
			return Sigmoid(Inputs * Weights - Bias);
		}

		double Sigmoid(double z)
		{
			return 1 / (1 + Math.Exp(-z));
		}

	}
}
