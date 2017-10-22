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
		public double mse { get; private set; }
		public int Epoch { get; private set; }

		public double LearnRate { get; private set; }

		public double[] oGrads;
		public double[] hGrads;
		#endregion

		internal void ShowNeuralNetwork()
		{
			// Show input values.
			if (numHidden == 0)
				Debug.Write($"Inputs: {outputNeurons[0].Inputs} W = {outputNeurons[0].Weights} ");
			else
				Debug.WriteLine($"Inputs: {hiddenNeurons[0].Inputs}");
			for (int k = 0; k < numOutputs; k++)
			{
				for (int j = 0; j < numHidden; j++)
				{
					Debug.WriteLine($"Weights {hiddenNeurons[j].Weights} output H{j,-2}: {Hidden[j],6:f2} * {outputNeurons[k].Weights[j]:f2} hGrad {hGrads[j]:f2}");
				}
				Debug.WriteLine($"Output = {Output[k]:f2} Target = {ExpectedOutput:f2} ograd {oGrads[k]:f2} learnRate {LearnRate} epoch {Epoch} mse = {mse:f2}");
			}
			Debug.WriteIf(mse >= 100, "NO CONVERGENCE ");
		}
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
			for (int i = 0; i < numHidden; i++) this.hiddenNeurons.Add(new Neuron(inputs: this.Inputs));
			for (int i = 0; i < numOutputs; i++) this.outputNeurons.Add(new Neuron(inputs: (numHidden > 0) ? Hidden : Inputs));
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
			this.LearnRate = learnRate;
			mse = 1.0;
			Epoch = 0;
			var idx = 0;
			while (mse > 0.01 && mse < 100 && ++Epoch < maxEpochs && (Inputs.Length + idx) < trainData.Length)
			{
				this.Inputs = SetInputs(trainData, idx);
				this.ExpectedOutput = trainData[numInputs + idx];
				ComputeOutputs();
				this.mse = Math.Abs(ExpectedOutput - Output[0]);
				BackPropagation(this.ExpectedOutput, learnRate);// Compute gradients and update Weights;
				// ++idx;
			}
			return mse;
		}

		/// <summary>
		/// Compute gradients and update weights and biases using back-propagation.
		/// 
		/// Usage index i,j,k:
		/// i = index Input neurons.
		/// j = index Hidden neurons.
		/// k = index Output neurons.
		/// </summary>
		private void BackPropagation(double expectedOutput, double learnRate)
		{
			ComputeGradients();
			ShowNeuralNetwork();
			UpdateWeights(learnRate);
		}

		private void UpdateWeights(double learnRate)
		{

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
			// 4. update output weights
			for (int k = 0; k < numOutputs; ++k)
			{
				for (int j = 0; j < outputNeurons[k].Inputs.Length; ++j)
				{
					// see above: hOutputs are inputs to the nn outputs
					// Divide output gradients over all inputs.
					double delta = learnRate * oGrads[k] / outputNeurons[k].Inputs[j] / numInputs;
					outputNeurons[k].Weights[j] += delta;
				}
			}
		}

		private void ComputeGradients()
		{
			//
			// 1. compute output gradients
			//
			oGrads = new Double[numOutputs];
			hGrads = new Double[numHidden];
			for (int k = 0; k < numOutputs; ++k)
			{
				//For sigmoid activation, the derivative of y = log-sigmoid(x) is y * (1 - y)
				double derivative = 1;// Output[k] * (1 - Output[k]);
									  // 'mean squared error version' includes (1-y)(y) derivative
				oGrads[k] = derivative * (ExpectedOutput - Output[k]);
			}
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