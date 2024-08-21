#ifndef OPTIM_H
#define OPTIM_H

#include <vector>
#include <optional>

#include "Tensor.h"
#include "TensorFactory.h"
#include "Slice.h"

namespace optim {

template <num::num_t T, typename Derived>
class OptimBase {
public:
	void zeroGradient()
	{
		static_cast<Derived *>(this)->zeroGradient();
	}

	void step()
	{
		static_cast<Derived *>(this)->step();
	}
};

template <num::num_t T>
class SGD : public OptimBase<T, SGD<T>> {
public:
	SGD(const std::vector<num::Tensor<T>>& parameters,
		 double learningRate = 0.1,
		 double momentum = 0.0)
	: parameters (parameters),
	  paramMomentum (parameters),
	  learningRate (learningRate),
	  momentum (momentum)
	{
		zeroGradient();
	}

	void zeroGradient()
	{
		for (int i = 0; i < parameters.size(); ++i) {
			parameters[i].zeroGradient();
			paramMomentum[i] = num::zeros<T>(parameters[i].dims);
		}
	}

	void step()
	{
		for (int i = 0; i < parameters.size(); ++i) {
			// std::cout << parameters[i].getGradient().toString() << std::endl;
			num::Tensor<T> paramUpdate = momentum * paramMomentum[i] -
					learningRate * parameters[i].getGradient();
			paramMomentum[i] = paramUpdate;
			parameters[i].set(parameters[i] + paramUpdate, {num::Slice{std::nullopt, std::nullopt, std::nullopt}});
		}
	}
private:
	std::vector<num::Tensor<T>> parameters;
	std::vector<num::Tensor<T>> paramMomentum;

	num::Tensor<T> learningRate;
	num::Tensor<T> momentum;
};



template <num::num_t T>
class Adam : public OptimBase<T, Adam<T>> {
public:
	Adam(const std::vector<num::Tensor<T>>& parameters,
		 double learningRate = 0.001,
		 double epsilon = 1e-7,
		 double beta_1 = 0.9,
		 double beta_2 = 0.999)
	: parameters (parameters),
	  paramMomentum (parameters),
	  paramCache (parameters),
	  learningRate (learningRate),
	  epsilon (epsilon),
	  beta_1 (beta_1),
	  beta_2 (beta_2),
	  iteration (1)
	{
		zeroGradient();
	}

	void zeroGradient()
	{
		for (int i = 0; i < parameters.size(); ++i) {
			parameters[i].zeroGradient();
			paramMomentum[i] = num::zeros<T>(parameters[i].dims);
			paramCache[i] = num::zeros<T>(parameters[i].dims);
		}
	}

	void step()
	{
		for (int i = 0; i < parameters.size(); ++i) {
			// std::cout << parameters[i].getGradient().toString() << std::endl;
			num::Tensor<T> grad = parameters[i].getGradient();
			paramMomentum[i] = beta_1 * paramMomentum[i] + (num::Tensor<T>(1) - beta_1) * grad;
			num::Tensor<T> momentumCorrected = paramMomentum[i] / (num::Tensor<T>(1) - autofn::pow<T>(beta_1, iteration));
			paramCache[i] = beta_2 * paramCache[i] + (num::Tensor<T>(1) - beta_2) * autofn::pow<T>(grad, 2);
			num::Tensor<T> cacheCorrected = paramCache[i] / (num::Tensor<T>(1) - autofn::pow<T>(beta_2, iteration));
			num::Tensor<T> paramUpdate = - learningRate * momentumCorrected /
					(autofn::pow<T>(cacheCorrected, 0.5) + epsilon);
			parameters[i].set(parameters[i] + paramUpdate, {num::Slice{std::nullopt, std::nullopt, std::nullopt}});
		}
		iteration += 1.0;
	}
private:
	std::vector<num::Tensor<T>> parameters;
	std::vector<num::Tensor<T>> paramMomentum;
	std::vector<num::Tensor<T>> paramCache;

	num::Tensor<T> learningRate;
	num::Tensor<T> epsilon;
	num::Tensor<T> beta_1;
	num::Tensor<T> beta_2;
	double iteration;
};



} // namespace optim

#endif