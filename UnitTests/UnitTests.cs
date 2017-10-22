using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Tests
{
	[TestClass()]
	public class NeuralNetworkTests
	{
		private int numInputs = 4;
		private int numHidden = 5;
		private int numOutputs = 6;

		[TestMethod()]
		public void TestNeuralNetworkTopology()
		{
			var nn = new NeuralNetwork(numInputs, numHidden, numOutputs);
			var I = nn.hiddenNeurons[0].Inputs.Length;
			var W = nn.hiddenNeurons[0].Weights.Length;
			var O = nn.outputNeurons[0].Inputs.Length;
			Assert.IsTrue(numInputs == I, $"Hidden layer has wrong number of inputs.{I}");
			Assert.IsTrue(numInputs == W, $"Hidden layer has wrong number of weights.{W}");
			Assert.IsTrue(numHidden == O, $"Output layer has wrong number of inputs.{O}");
			//
			// Test if hidden neuron is connected to output neuron.
			//
			nn.Hidden[0] = 123; // set hidden output to arbitrary value.
			Assert.IsTrue(nn.Hidden[0] == nn.outputNeurons[1].Inputs[0], $"Hidden neuron is not connected to output neuron.");
			nn.Hidden[numHidden - 1] = 456; // set last hidden output to arbitrary value.
			Assert.IsTrue(nn.Hidden[numHidden - 1] == nn.outputNeurons[1].Inputs[numHidden - 1], $"Last hidden neuron is not connected to last input of output neuron.");
			//
			// Test if input neuron is connected to output neuron when no hidden layer.
			//
			nn = new NeuralNetwork(4, 0, 2);
			nn.Inputs[3] = 123; // set input to arbitrary value.
			Assert.IsTrue(nn.Inputs[3] == nn.outputNeurons[1].Inputs[3], $"Input neuron not connected to output neuron when no hidden layer.");
		}
		[TestMethod()]
		public void TestSingleNeuron()
		{
			//
			// Neural network: Single output neuron with one input.
			//
			var nn = new NeuralNetwork(numInputs: 1, numHidden: 0, numOutputs: 1);
			var mse = nn.Train(trainData: new double[] { 1, 1 }, maxEpochs: 100, learnRate: 1);
			Assert.IsTrue(mse < 0.01 && nn.Epoch == 2, $"Error = {mse} epoch {nn.Epoch}");

			nn = new NeuralNetwork(numInputs: 1, numHidden: 0, numOutputs: 1);
			mse = nn.Train(new double[] { 6, 1 }, 100, 1);
			Assert.IsTrue(mse < 0.01 && nn.Epoch == 2, $"Error = {mse} epoch {nn.Epoch} input 6");
		}
		[TestMethod()]
		public void TestSingleNeuronTwoInputs()
		{
			//
			// Neural network: Single output neuron with two input.
			//
			var nn = new NeuralNetwork(numInputs: 2, numHidden: 0, numOutputs: 1);
			var mse = nn.Train(trainData: new double[] { 1, 1, 1 }, maxEpochs: 100, learnRate: 1);
			Assert.IsTrue(mse < 0.01 && nn.Epoch == 2, $"Error = {mse} epoch {nn.Epoch}");
		}
		[TestMethod()]
		public void TestSingleNeuronTenInputs()
		{
			//
			// Neural network: Single output neuron with ten input.
			//
			var nn = new NeuralNetwork(numInputs: 10, numHidden: 0, numOutputs: 1);
			var mse = nn.Train(trainData: new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1 }, maxEpochs: 100, learnRate: 1);
			Assert.IsTrue(mse < 0.01 && nn.Epoch == 2, $"Error = {mse} epoch {nn.Epoch}");
		}
		[TestMethod()]
		public void TestHiddenNeuron()
		{
			//
			// Neural network: Single hidden neuron with one input.
			//
			var nn = new NeuralNetwork(numInputs: 1, numHidden: 1, numOutputs: 1);
			var mse = nn.Train(trainData: new double[] { 1, 1 }, maxEpochs: 100, learnRate: 0.8);
			Assert.IsTrue(mse < 0.01, $"Error Hidden neuron= {mse:f0}");

		}
		[TestMethod()]
		public void TestNandNeuron()
		{
			//
			// Neural network: Nand 2 inputs and 2 hidden neuron.
			//
			var nn = new NeuralNetwork(numInputs: 2, numHidden: 2, numOutputs: 1);
			var mse = 1.0;
			mse = nn.Train(trainData: new double[] { 0,0,1 }, maxEpochs: 10, learnRate: 0.25);
			mse = nn.Train(trainData: new double[] { 0,1,1 }, maxEpochs: 10, learnRate: 0.25);
			mse = nn.Train(trainData: new double[] { 1,0,1 }, maxEpochs: 10, learnRate: 0.25);
			mse = nn.Train(trainData: new double[] { 1,1,0 }, maxEpochs: 10, learnRate: 0.25);
			Assert.IsTrue(mse < 0.01, $"Error Nand neuron= {mse:f2}");

		}
	}
}