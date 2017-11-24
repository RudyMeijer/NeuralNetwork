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
        /// <summary>
        /// Next two properties must be readonly!
        /// Because all constructed neurons have a reference to these input properties.
        /// Therefore you can't change reference by Inputs = new Vector(...)
        /// Use SetInputs methode instead. Which assign vectoritems directly by Inputs[0] = ...
        /// </summary>
        private Vector inputs;

        public Vector Inputs
        {
            get => inputs;
            set { for (int i = 0; i < inputs.Length; i++) inputs[i] = value[i]; }
        }

        public Vector Hidden { get; }
        public Vector Output;
        public double ExpectedOutput;
        public List<Neuron> hiddenNeurons = new List<Neuron>();
        public List<Neuron> outputNeurons = new List<Neuron>();
        public double mse { get; private set; }
        public int Epoch { get; set; }

        public double LearnRate { get; private set; }

        public Vector oLast;
        public Vector oGrads;
        public Vector hGrads;
        #endregion

        public NeuralNetwork(int numInputs, int numHidden, int numOutputs)
        {
            this.numInputs = numInputs;
            this.numHidden = numHidden;
            this.numOutputs = numOutputs;
            this.inputs = new Vector(numInputs);
            this.Hidden = new Vector(numHidden);
            this.Output = new Vector(numOutputs);
            hiddenNeurons.Clear();
            outputNeurons.Clear();
            for (int i = 0; i < numHidden; i++) this.hiddenNeurons.Add(new Neuron(inputs: this.Inputs));
            for (int i = 0; i < numOutputs; i++) this.outputNeurons.Add(new Neuron(inputs: (numHidden == 0) ? Inputs : Hidden));
            // Output and hidden Gradients.
            this.oLast = new Vector(numOutputs);
            this.oGrads = new Vector(numOutputs);
            this.hGrads = new Vector(numHidden);
        }
        public double Train(double[] trainData, int maxEpochs, double learnRate)
        {
            this.LearnRate = learnRate;
            mse = 1.0;
            Epoch = 0;
            oLast.Clear();
            var idx = 0;
            while (mse > 0.01 && mse < 100 && ++Epoch <= maxEpochs) // && (Inputs.Length + idx) < trainData.Length) // for time related datastreams
            {
                SetInputs(trainData, idx);
                this.ExpectedOutput = trainData[numInputs + idx];
                ComputeOutputs();
                this.mse = Math.Abs(ExpectedOutput - Output[0]);
                BackPropagation(this.ExpectedOutput, this.LearnRate);// Compute gradients and update Weights;
                                                                     //idx += 1; //if (idx >= trainData.Length) idx = 0;
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
            Program.ShowNeuralNetwork(this);
            UpdateWeights(this.LearnRate);
        }
        private void ComputeGradients()
        {
            //
            // 1. compute output gradients
            //
            for (int k = 0; k < numOutputs; ++k)
            {
                //For sigmoid activation, the derivative of y = log-sigmoid(x) is y * (1 - y)
                // when output = 1 then derivative = 0 and ograds = 0 althow error = -0.9 (T-O = 0.1-1) 
                double derivative = 1;// Output[k] * (1 - Output[k]);
                                      // 'mean squared error version' includes (1-y)(y) derivative
                oGrads[k] = derivative * (ExpectedOutput - Output[k]);
                // 
                // If gradient switches sign then half learningrate.
                //
                if (oGrads[k] * oLast[k] < 0) this.LearnRate /= 2;
                oLast[k] = oGrads[k];
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
            #region BIAS
            // 3b. update hidden biases
            for (int j = 0; j < numHidden; ++j)
            {
                double delta = learnRate * hGrads[j] * 1.0; // t1.0 is constant input for bias; could leave out
                hiddenNeurons[j].Bias += delta;
            }
            #endregion
            // 4. update output weights
            for (int k = 0; k < numOutputs; ++k)
            {
                for (int j = 0; j < outputNeurons[k].Inputs.Length; ++j)
                {
                    // see above: hOutputs are inputs to the nn outputs
                    // Divide output gradients over all inputs.
                    //double delta = learnRate * oGrads[k] / outputNeurons[k].Inputs[j] / numInputs;
                    double delta = learnRate * oGrads[k] / outputNeurons[k].Inputs.Length;
                    outputNeurons[k].Weights[j] += delta;
                }
            }
        }
        public Vector ComputeOutputs()
        {
            for (int i = 0; i < numHidden; i++) this.Hidden[i] = hiddenNeurons[i].Execute();
            for (int i = 0; i < numOutputs; i++) this.Output[i] = outputNeurons[i].Execute();
            return Output;
        }
        public Vector SetInputs(double[] trainData, int idx)
        {
            for (int i = 0; i < Inputs.Length; i++)
            {
                Inputs[i] = trainData[idx + i];
                if (Inputs[i] == 0) Inputs[i] = 0.01; // Inputs mogen niet null zijn (delta W = ~)
            }
            return Inputs;
        }
    }
}