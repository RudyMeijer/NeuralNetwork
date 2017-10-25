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
		static Random random = new Random();
		public Neuron(Vector inputs)
		{
			this.Inputs = inputs;
			this.Weights = new Vector(Inputs.Length);
			this.Bias = random.NextDouble();
			for (int i = 0; i < Inputs.Length; i++) this.Weights[i] = random.NextDouble();
		}

		public double Execute() => Sigmoid(Inputs * Weights);// - Bias);

		public double Sigmoid(double z) => 1 / (1 + Math.Exp(-8 * z));
		public static double SigmoidInv(double z) => -Math.Log(1 / z - 1) / 8; // Used for display only.

		private static double ATanh(double z) => (Math.Log(1 + z) - Math.Log(1 - z)) / 2;
	}
}
