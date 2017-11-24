using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NeuralNetwork;
using log = NeuralNetwork.Program;

namespace WinForms
{
    public partial class Form1 : Form
    {
        //private int idx;
        private Binding b;
        private NeuralNetwork.NeuralNetwork nn;

        public Form1()
        {
            InitializeComponent();
            //
            // Write all Debug output to a textbox.
            //
            Debug.Listeners.Add(new ControlBoxWriter(textBox1));
            NeuralNetwork_Click(null, null);
        }
        #region BIND
        private void Bind(NumericUpDown control, string propertyName, string dataMember)
        {
            b = new Binding(propertyName, this, dataMember);
            control.DataBindings.Add(b);
            //b.FormattingEnabled = true;
            b.Parse += B_Parse;
        }
        /// <summary>
        /// Dit event treedt op wanneer huidige control is gewijzigd en uit focus raakt.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void B_Parse(object sender, ConvertEventArgs e)
        {
            var c = (sender as Binding).Control as NumericUpDown;
            var n = $"sender={c.Name} value={c.Value}"; //Werkt!!
            button1_Click(sender, null);
        }
        #endregion

        public int numInputs { get; set; }
        public int numHidden { get; set; }
        public int numOutputs { get; set; }
        public int maxEpochs { get; set; }
        public double learningRate { get; set; }

        private void button1_Click(object sender, EventArgs e)
        {
            textBox1.Clear();
            nn.Train(null, maxEpochs, learningRate);
            Debug.Write("===== end ========");
        }

        private void chkAuto_CheckedChanged(object sender, EventArgs e)
        {
            timer1.Enabled = chkAuto.Checked;
        }

        private void NeuralNetwork_Click(object sender, EventArgs e)
        {
            numInputs = (int)numericUpDown1.Value;
            numHidden = (int)numericUpDown2.Value;
            numOutputs = (int)numericUpDown3.Value;
            nn = new NeuralNetwork.NeuralNetwork(numInputs, numHidden, numOutputs);
            TrainingData_Click(null, null);
        }

        private void TrainingData_Click(object sender, EventArgs e)
        {
            maxEpochs = (int)numericUpDown4.Value;
            learningRate = (double)numericUpDown5.Value;
            var inputValues = (double)numericUpDown6.Value;
            nn.Inputs = new Lib.Vector(numInputs, inputValues);
            nn.ExpectedOutput = (double)numericUpDown7.Value;
            button1_Click(null, null);
        }
    }
}
