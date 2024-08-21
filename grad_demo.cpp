#include <iostream>

#include "autograd/autograd.h"

int main()
{
	num::Tensor<double> a(-4);
	num::Tensor<double> b(2);
	num::Tensor<double> c = a + b;
	
	num::Tensor<double> d = a * b + autofn::pow<double>(b, 3);
	c = c + c + num::Tensor<double>(1);
	c = c + num::Tensor<double>(1) + c + (-a);
	d = d + d * num::Tensor<double>(2) + autofn::relu<double>(b + a);
	d = d + num::Tensor<double>(3) * d + autofn::relu<double>(b - a);
	num::Tensor<double> e = c - d;
	num::Tensor<double> f = autofn::pow<double>(e, 2);
	num::Tensor<double> g = f / num::Tensor<double>(2);
	g = g + num::Tensor<double>(10)/f;
	std::cout << "g: " << g.toString() << std::endl; // prints 24.7041, the outcome of this forward pass
	g.backward();
	// prints 138.8338, i.e. the numerical value of dg/da
	std::cout << "a gradient: " << a.getGradient().toString() << std::endl;
	// prints 645.5773, i.e. the numerical value of dg/db
	std::cout << "b gradient: " << b.getGradient().toString() << std::endl;
}