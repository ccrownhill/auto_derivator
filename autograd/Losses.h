#ifndef LOSSES_H
#define LOSSES_H

#include <optional>
#include "AutogradFunction.h"

namespace autofn {

template <num::num_t T>
class MSELoss : public Function<T, MSELoss<T>> {
public:
	static num::Tensor<T> operator()(const num::Tensor<T>& a, const num::Tensor<T>& b)
	{
		return Function<T, MSELoss<T>>::apply({a, b});
	}

	static num::Tensor<T> forward(const std::vector<num::Tensor<T>>& args)
	{
		if (args.size() != 2) {
			throw std::invalid_argument("MSELoss needs exactly 2 operands");
		}

		return (args.at(0) - args.at(1)).pow_(2);
	}

	static void backward(const num::Tensor<T>& outGradient, std::vector<num::Tensor<T>> oldInputs)
	{
		num::Tensor<T> gradient = outGradient * num::Tensor<T>(2) * (oldInputs[0] - oldInputs[1]);
		oldInputs[0].setBroadcastGradient(gradient);
		oldInputs[1].setBroadcastGradient(num::Tensor<T>(-1) * gradient);
	}
};

template <num::num_t T>
inline constexpr MSELoss<T> mseLoss{};

} // namespace autofn

#endif