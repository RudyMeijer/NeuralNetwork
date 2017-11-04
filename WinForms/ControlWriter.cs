using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WinForms
{
    /// <summary>
    /// This class writes Debug output to a window forms control.
    /// 
    /// Debug.Listeners.Add(new ControlBoxWriter(textBox1));
    /// </summary>
    public class ControlBoxWriter : TextWriterTraceListener
    {
        private Control textbox;
        public ControlBoxWriter(Control textbox)
        {
            this.textbox = textbox;
        }

        public override void Write(string value)
        {
            textbox.Text += value;
        }

        public override void WriteLine(string value)
        {
            textbox.Text += value + "\r\n";
        }
    }
    /// <summary>
    /// Console.SetOut(new ControlWriter(textBox1));
    /// </summary>
    public class ControlWriter : TextWriter
    {
        private Control textbox;
        public ControlWriter(Control textbox)
        {
            this.textbox = textbox;
        }

        public override void Write(string value)
        {
            textbox.Text += value;
        }

        public override void WriteLine(string value)
        {
            textbox.Text += value + "\r\n";
        }

        public override Encoding Encoding
        {
            get { return Encoding.ASCII; }
        }
    }
    //Bind(numericUpDown1, "Value", nameof( numInputs));
    //numericUpDown1.DataBindings.Add("Value", this, nameof(numInputs));
    //numericUpDown2.DataBindings.Add("Value", this, nameof(numHidden));
    //numericUpDown3.DataBindings.Add("Value", this, nameof(numOutputs));
    //numericUpDown4.DataBindings.Add("Value", this, nameof(maxEpochs));
    //numericUpDown5.DataBindings.Add("Value", this, nameof(learningRate));
    //numericUpDown6.DataBindings.Add("Value", this, nameof(inputValues));
    //numericUpDown7.DataBindings.Add("Value", this, nameof(targetValue));
}
