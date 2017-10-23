using System;
using System.Text;

namespace Lib
{
	/// <summary>
	/// Description of Vector.
	/// </summary>
	public class Vector
	{
		private readonly double[] items;
		public int Length { get { return items.Length; } }
		public delegate double funct(int r);
		/// <summary>
		/// Initialize a vector of length items and fill them with a default value.
		/// </summary>
		/// <param name="length"></param>
		/// <param name="value"></param>
		public Vector(int length, double defaultValue = 0)
		{
			items = new double[length];
			for (int i = 0; i < length; i++) items[i] = defaultValue;
		}
		public Vector(params double[] args)
		{
			items = args;
		}
		public static double operator *(Vector v1, Vector v2)
		{
			var dot = 0d;
			for (int i = 0; i < v1.Length; i++) dot += v1[i] * v2[i];
			return dot;
		}
		public double this[int i]
		{
			get
			{
				return items[i];
			}
			set
			{
				items[i] = value;
			}
		}
		public override string ToString()
		{
			var sb = new StringBuilder();
			for (int i = 0; i < Length; i++) sb.Append($"{items[i],6:f2}");
			return sb.ToString();
		}

		internal void Clear()
		{
			for (int i = 0; i < this.Length; i++) items[i] = 0;
		}
	}
}
