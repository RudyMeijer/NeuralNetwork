using Lib;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace NeuralNetwork
{
	public class NeuralNetwork
	{
		#region FIELDS
		public int numInputs;
		public int numHidden;
		public int numOutputs;
		public Vector Inputs;
		public Vector Hidden;
		public Vector Output;
		public double ExpectedOutput;
		public List<Neuron> hiddenNeurons = new List<Neuron>();
		public List<Neuron> outputNeurons = new List<Neuron>();
		public double mse;
		public int epoch;
		public int idx;
		#endregion
		/// <summary>
		/// Create Neural network with Hidden- and Output neurons.
		/// </summary>
		/// <param name="numInputs"></param>
		/// <param name="numHidden"></param>
		/// <param name="numOutputs"></param>
		public NeuralNetwork(int numInputs, int numHidden, int numOutputs)
		{
			this.numInputs = numInputs;
			this.numHidden = numHidden;
			this.numOutputs = numOutputs;
			this.Inputs = new Vector(numInputs);
			this.Hidden = new Vector(numHidden);
			this.Output = new Vector(numOutputs);
			hiddenNeurons.Clear();
			outputNeurons.Clear();
			for (int i = 0; i < numHidden; i++) this.hiddenNeurons.Add(new Neuron(this.Inputs));
			for (int i = 0; i < numOutputs; i++) this.outputNeurons.Add(new Neuron((numHidden > 0) ? Hidden : Inputs));
		}
		/// <summary>
		/// Train the neural network with train data.
		/// </summary>
		/// <param name="trainData"></param>
		/// <param name="maxEpochs"></param>
		/// <param name="learnRate"></param>
		/// <returns></returns>
		public double Train(double[] trainData, int maxEpochs, double learnRate)
		{
			mse = 1.0;
			epoch = 0;
			idx = 0;
			while (epoch < maxEpochs && mse > 0.01 && mse < 100 && (Inputs.Length + idx) < trainData.Length)
			{
				this.Inputs = SetInputs(trainData, idx);
				this.ExpectedOutput = trainData[numInputs + idx];
				ComputeOutputs();
				mse = Math.Abs(ExpectedOutput-Output[0]);
				Program.ShowNeuralNetwork(this);
				BackPropagation(this.ExpectedOutput, learnRate);// Update Weights;
				++epoch; //++idx;
			}
			return mse;
		}

		/// <summary>
		/// Update the weights and biases using back-propagation.
		/// Index: i=Input, j=Hidden and k=Output neurons. 
		/// </summary>
		private void BackPropagation(double expectedOutput, double learnRate)
		{
			//
			// 1. compute output gradients
			//
			var oGrads = new Double[numOutputs];
			var hGrads = new Double[numHidden];
			for (int k = 0; k < numOutputs; ++k)
			{
				//For sigmoid activation, the derivative of y = log-sigmoid(x) is y * (1 - y)
				double derivative = 1;// Output[k] * (1 - Output[k]);
				// 'mean squared error version' includes (1-y)(y) derivative
				oGrads[k] = derivative * (ExpectedOutput - Output[k]);
			}
			#region HIDDEN
			//
			// 2. compute hidden gradients
			//
			for (int j = 0; j < numHidden; ++j)
			{
				// derivative of tanh = (1 - y) * (1 + y)
				double derivative = 1;// (1 - Hidden[j]) * (1 + Hidden[j]);
				double sum = 0.0;
				for (int k = 0; k < numOutputs; ++k) // each hidden delta is the sum of numOutput terms
				{
					double x = oGrads[k] * outputNeurons[k].Weights[j];
					sum += x;
				}
				hGrads[j] = derivative * sum;
			}

			// 3a. update hidden weights (gradients must be computed right-to-left but weights
			// can be updated in any order)
			for (int j = 0; j < numHidden; ++j)
			{
				for (int i = 0; i < numInputs; ++i)
				{
					double delta = learnRate * hGrads[j] * hiddenNeurons[j].Inputs[i]; // compute the new delta
					hiddenNeurons[j].Weights[i] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
				}
			}

			// 3b. update hidden biases
			for (int j = 0; j < numHidden; ++j)
			{
				double delta = learnRate * hGrads[j] * 1.0; // t1.0 is constant input for bias; could leave out
				hiddenNeurons[j].Bias += delta;
			}
			#endregion
			// 4. update hidden-output weights
			for (int k = 0; k < numOutputs; ++k)
			{
				for (int j = 0; j < outputNeurons[k].Inputs.Length; ++j)
				{
					// see above: hOutputs are inputs to the nn outputs
					double delta = learnRate * oGrads[k] / outputNeurons[k].Inputs[j];
					outputNeurons[k].Weights[j] += delta;
				}
			}
		}

		private Vector ComputeOutputs()
		{
			for (int i = 0; i < numHidden; i++) this.Hidden[i] = hiddenNeurons[i].ExecuteIW();
			for (int i = 0; i < numOutputs; i++) this.Output[i] = outputNeurons[i].ExecuteIW();
			return Output;
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