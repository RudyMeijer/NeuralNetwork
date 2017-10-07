using System;

namespace Lib
{
	/// <summary>
	/// Description of Vector.
	/// </summary>
	public class Vector
	{
		private readonly double[] items;
		public int Length { get { return items.Length; } }
		public Vector(params double[] args)
		{
			items = args;
		}
		public static double operator*(Vector v1, Vector v2)
		{
			var dot = 0d;
			for (int i = 0; i < v1.Length; i++) dot += v1[i] * v2[i];
			return dot;
		}
		public double this[int i] {
			get {
				return items[i];
			}
			set {
				items[i] = value;
			}
		}
		public override string ToString()
		{
			var s = "";
			for (int i = 0; i < Length; i++)
			{
				s += ((i==0)?"":", ")+items[i].ToString();
			}
			return s;
		}
	}
}
