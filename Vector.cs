using System;
using System.Text;

namespace Lib
{
	/// <summary>
	/// This Vector class defines an array of doubles.
	/// And allow vector operations * /
	/// </summary>
	public class Vector
	{
		private readonly double[] items;
		public int Length { get { return items.Length; } }
		/// <summary>
		/// Initialize a vector of length items and fill them with a default value.
		/// </summary>
		/// <param name="length">Vector length</param>
		/// <param name="value">Value of each vector item</param>
		#region CONSTRUCTORS
		public Vector(int length, double defaultValue = 0)
		{
			items = new double[length];
			for (int i = 0; i < length; i++) items[i] = defaultValue;
		}
		public Vector(params double[] args)
		{
			items = args;
		}
		#endregion
		#region operators
		public static double operator *(Vector v1, Vector v2)
		{
			var dot = 0d;
			for (int i = 0; i < v1.Length; i++) dot += v1[i] * v2[i];
			return dot;
		}
		public static Vector operator /(Double dot, Vector v1)
		{
			var prod = dot / v1.Length;
			Vector v2 = new Vector(v1.Length);
			for (int i = 0; i < v1.Length; i++) v2[i] = prod / v1[i];
			return v2;
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
		#endregion
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
