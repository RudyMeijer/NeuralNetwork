using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Lib;

namespace UnitTests
{
	[TestClass]
	public class UnitTestVector
	{
		[TestMethod]
		public void TestDivide()
		{
			var product = 10;
			var v1 = new Vector(1, 2, 3);
			var v2 = product / v1;
			var answer = v1 * v2;
			Assert.IsTrue(answer == product, $"Error Vector divide {answer})");
		}
	}
}

