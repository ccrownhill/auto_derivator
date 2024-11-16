#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include "Tensor.h"

namespace nn {

template <num::num_t T, typename Derived>
class Module {
public:
	std::vector<num::Tensor<T>> parameters;

	num::Tensor<T> forward(const num::Tensor<T>& x) const
	{
		return static_cast<Derived*>(this)->forward(x);
	}

	num::Tensor<T> registerParameter(const num::Tensor<T>& parameter)
	{
		parameters.push_back(parameter);
		return parameter;
	}
	template <typename modelT>
	modelT registerModule(const modelT& module)
	{
		for (const auto& p : module.parameters) {
			parameters.push_back(p);
		}
		return module;
	}
};

template <num::num_t T>
class Linear : public Module<T, Linear<T>> {
public:
	num::Tensor<T> w;
	num::Tensor<T> b;

	// weights multiplied by 0.1 to keep them very small
	Linear(int inFeatures, int outFeatures, bool withBias = true)
	: w (this->registerParameter(num::Tensor<T>(0.1) * num::randn<T>({outFeatures, inFeatures}))),
	  b (num::zeros<T>({outFeatures})),
	  withBias (withBias)
	{
		if (withBias) {
			this->registerParameter(b);
		}
	}

	num::Tensor<T> forward(const num::Tensor<T>& x) const
	{
        std::cout << "x: " << x.toString() << "; w: " << w.toString() << std::endl;
		return autofn::mm<T>(x, autofn::transpose<T>(w)) + b;
	}
private:
	bool withBias;
};

} // namespace nn

#endif
