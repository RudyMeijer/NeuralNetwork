using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using NeuralNetwork;
using WpfTutorialSamples.DataBinding;
using WpfTutorialSamples;
namespace Wpf
{
	/// <summary>
	/// Interaction logic for MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window
	{
		public int NumInputs { get; set; }
		public int NumHidden { get; set; }
		public int NumOutputs { get; set; }
		public MainWindow()
		{
			InitializeComponent();
			this.DataContext = this;
			NumInputs = 1;
			NumHidden = 2;
			NumOutputs = 3;
		}


		private void btnStart_Click(object sender, RoutedEventArgs e)
		{
			NumInputs++;
			//txtNumInputs.Text = NumInputs.ToString();
			//var nn = new NeuralNetwork.NeuralNetwork(NumInputs, NumHidden, NumOutputs);
		}

		private void button_Click(object sender, RoutedEventArgs e)
		{
			var x = new DataContextSample();
			x.Show();
		}
	}
}
