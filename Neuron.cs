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
		public Neuron(Vector inputs)
		{
			this.Inputs = inputs;
			this.Weights = new Vector(this.Inputs.Length);
			var random = new Random();
			this.Bias = random.NextDouble();
			for (int i = 0; i < this.Inputs.Length; i++) this.Weights[i] = random.NextDouble();
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
