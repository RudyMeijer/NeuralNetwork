using Lib;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Tests
{
	[TestClass()]
	public class NeuralNetworkTests
	{
		//private int numInputs = 4;
		//private int numHidden = 5;
		//private int numOutputs = 6;

		[TestMethod()]
		public void TestNeuralNetworkTopology()
		{
			var nn = new NeuralNetwork(numInputs: 4, numHidden: 5, numOutputs: 6);
			var I = nn.hiddenNeurons[0].Inputs.Length;
			var W = nn.hiddenNeurons[4].Weights.Length;
			var O = nn.outputNeurons[5].Inputs.Length;
			Assert.IsTrue(4 == I, $"Hidden layer has wrong number of inputs.{I}");
			Assert.IsTrue(4 == W, $"Hidden layer has wrong number of weights.{W}");
			Assert.IsTrue(5 == O, $"Output layer has wrong number of inputs.{O}");
			//
			// Test if hidden neuron is connected to output neuron.
			//
			nn.Hidden[0] = 123; // set hidden output to arbitrary value.
			Assert.IsTrue(nn.Hidden[0] == nn.outputNeurons[1].Inputs[0], $"Hidden neuron is not connected to output neuron.");
			nn.Hidden[4] = 456; // set last hidden output to arbitrary value.
			Assert.IsTrue(nn.Hidden[4] == nn.outputNeurons[1].Inputs[4], $"Last hidden neuron is not connected to last input of output neuron.");
			//
			// Test if input neuron is connected to output neuron when no hidden layer.
			//
			nn = new NeuralNetwork(4, 0, 2);
			nn.Inputs[0] = 123; // set first input to arbitrary value.
			nn.Inputs[3] = 234; // set last input to arbitrary value.
			Assert.IsTrue(123 == nn.outputNeurons[0].Inputs[0], $"Input 0 neuron not connected to output neuron 0 when no hidden layer.");
			Assert.IsTrue(123 == nn.outputNeurons[1].Inputs[0], $"Input 0 neuron not connected to output neuron 1 when no hidden layer.");
			Assert.IsTrue(234 == nn.outputNeurons[0].Inputs[3], $"Input 3 neuron not connected to output neuron 0 when no hidden layer.");
			Assert.IsTrue(234 == nn.outputNeurons[1].Inputs[3], $"Input 3 neuron not connected to output neuron 1 when no hidden layer.");
			//
			// Check if Internal input of a single neuron equals input neuron.
			//
			nn.Inputs = new Vector(111, 222, 3, 4 );
			var inp = nn.outputNeurons[0].Inputs[0];
			Assert.IsTrue(inp == 111, $"Internal input of a single neuron not equal to input neural network.");
		}
		[TestMethod()]
		public void TestSingleNeuron()
		{
			//
			// Neural network: Single output neuron with one input.
			//
			var nn = new NeuralNetwork(numInputs: 1, numHidden: 0, numOutputs: 1);
			var mse = nn.Train(trainData: new double[] { 1, .5 }, maxEpochs: 100, learnRate: 10);
			Assert.IsTrue(mse < 0.01 && nn.Epoch <= 12, $"Error 1 = {mse} epoch {nn.Epoch}");

			nn = new NeuralNetwork(numInputs: 1, numHidden: 0, numOutputs: 1);
			mse = nn.Train(new double[] { 6, 1 }, 100, 10);
			Assert.IsTrue(mse < 0.01 && nn.Epoch == 1, $"Error 2 = {mse} epoch {nn.Epoch} input 6");
		}
		[TestMethod()]
		public void TestSingleNeuronTwoInputs()
		{
			//
			// Neural network: Single output neuron with two input.
			//
			var nn = new NeuralNetwork(numInputs: 2, numHidden: 0, numOutputs: 1);
			var mse = nn.Train(trainData: new double[] { 1, 1, 0.5 }, maxEpochs: 100, learnRate: 1);
			Assert.IsTrue(mse < 0.01 && nn.Epoch <= 8, $"Error = {mse} epoch {nn.Epoch}");
		}
		[TestMethod()]
		public void TestSingleNeuronTenInputs()
		{
			//
			// Neural network: Single output neuron with ten input.
			//
			var nn = new NeuralNetwork(numInputs: 10, numHidden: 0, numOutputs: 1);
			var mse = nn.Train(trainData: new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0.5 }, maxEpochs: 100, learnRate: 10);
			Assert.IsTrue(mse < 0.01 && nn.Epoch <= 18, $"Error = {mse} epoch {nn.Epoch}");
		}
		[TestMethod()]
		public void TestHiddenNeuron()
		{
			//
			// Neural network: Single hidden neuron with one input.
			//
			var nn = new NeuralNetwork(numInputs: 1, numHidden: 1, numOutputs: 1);
			var mse = nn.Train(trainData: new double[] { 1, 0.5 }, maxEpochs: 100, learnRate: 1);
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
			mse = nn.Train(trainData: new double[] { 0, 0, .5 }, maxEpochs: 100, learnRate: 1);
			mse = nn.Train(trainData: new double[] { 0, 1, .5 }, maxEpochs: 100, learnRate: 1);
			mse = nn.Train(trainData: new double[] { 1, 0, .5 }, maxEpochs: 100, learnRate: 1);
			mse = nn.Train(trainData: new double[] { 1, 1, 0 }, maxEpochs: 100, learnRate: 1);
			Assert.IsTrue(mse < 0.01, $"Error Nand neuron= {mse:f2}");
			//
			// Now Compute outputs for a given input combination.
			//
			nn.Inputs = new Vector(1d,1);
			var output = nn.ComputeOutputs()[0];
			Program.ShowNeuralNetwork(nn);
			Assert.IsTrue(output < 0.1, $"Nand 0 0 = {output}");
		}
		[TestMethod()]
		public void TestSigmoidInverse()
		{
			//
			// Create a Neuron to address it's Sigmoid function only.
			//
			var n = new Neuron(new Lib.Vector(1d, 1d));
			var e = 0.0001;
			for (int x = -10; x < 5; x++)
			{
				var s = n.Sigmoid(x);
				var y = Program.SigmoidInv(s);
				Assert.IsTrue(y >= x - e && y <= x + e, $"y not equal to x. y={y} x={x} s={s}");
			}
			var inf = Program.SigmoidInv(n.Sigmoid(5));
			Assert.IsTrue(Double.IsInfinity(inf), "SigmoidInv(5) is infinity. ");
		}
	}
}