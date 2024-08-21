#ifndef AUTOGRAD_FUNCTION_H
#define AUTOGRAD_FUNCTION_H

#include <initializer_list>
#include <stdexcept>

#include "Tensor.h"

namespace autofn {


template <num::num_t T, typename Derived>
class Function {
public:
	static num::Tensor<T> apply(std::initializer_list<num::Tensor<T>> args)
	{
		std::vector<num::Tensor<T>> inputs(args);
		num::Tensor<T> out = Derived::forward(inputs);
		out.gradGraphChildren = inputs;
		out.backwardFn = Derived::backward;
		return out;
	}
};


template <num::num_t T>
class Add : public Function<T, Add<T>> {
public:

	static num::Tensor<T> forward(const std::vector<num::Tensor<T>>& args)
	{
		if (args.size() != 2) {
			throw std::invalid_argument("add needs exactly 2 operands");
		}

		return num::applyBinaryWithBroadcast(
			args.at(0), args.at(1),
			[](T a, T b) -> T {
				return a + b;
			});
	}

	static void backward(const num::Tensor<T>& outGradient, std::vector<num::Tensor<T>> oldInputs)
	{
		oldInputs[0].setBroadcastGradient(outGradient);
		oldInputs[1].setBroadcastGradient(outGradient);
	}
};
template <num::num_t T>
class Sub : public Function<T, Sub<T>> {
public:
	static num::Tensor<T> forward(const std::vector<num::Tensor<T>>& args)
	{
		if (args.size() != 2) {
			throw std::invalid_argument("add needs exactly 2 operands");
		}

		return num::applyBinaryWithBroadcast(
			args.at(0), args.at(1),
			[](T a, T b) -> T {
				return a - b;
			});
	}

	static void backward(const num::Tensor<T>& outGradient, std::vector<num::Tensor<T>> oldInputs)
	{
		oldInputs[0].setBroadcastGradient(outGradient);
		oldInputs[1].setBroadcastGradient(num::Tensor<T>(-1) * outGradient);
	}
};
template <num::num_t T>
class Mul : public Function<T, Mul<T>> {
public:
	static num::Tensor<T> forward(const std::vector<num::Tensor<T>>& args)
	{
		if (args.size() != 2) {
			throw std::invalid_argument("add needs exactly 2 operands");
		}

		return num::applyBinaryWithBroadcast(
			args.at(0), args.at(1),
			[](T a, T b) -> T {
				return a * b;
			});
	}

	static void backward(const num::Tensor<T>& outGradient, std::vector<num::Tensor<T>> oldInputs)
	{
		oldInputs[0].setBroadcastGradient(oldInputs[1] * outGradient);
		oldInputs[1].setBroadcastGradient(oldInputs[0] * outGradient);
	}
};

template <num::num_t T>
class Div : public Function<T, Div<T>> {
public:
	static num::Tensor<T> forward(const std::vector<num::Tensor<T>>& args)
	{
		if (args.size() != 2) {
			throw std::invalid_argument("add needs exactly 2 operands");
		}


		return num::applyBinaryWithBroadcast(
			args.at(0), args.at(1),
			[](T a, T b) -> T {
				return a / b;
			});
	}

	static void backward(const num::Tensor<T>& outGradient, std::vector<num::Tensor<T>> oldInputs)
	{
		oldInputs[0].setBroadcastGradient(outGradient / oldInputs[1]);
		oldInputs[1].setBroadcastGradient((num::Tensor<T>(-1) * oldInputs[0]) * oldInputs[1].pow(-2) * outGradient);
	}
};

template <num::num_t T>
class Pow : public Function<T, Pow<T>> {
public:
	static num::Tensor<T> operator()(const num::Tensor<T>& base, double power)
	{
		return Function<T, Pow<T>>::apply({base, num::Tensor<T>(power)});
	}

	static num::Tensor<T> operator()(const num::Tensor<T>& base, const num::Tensor<T>& power)
	{
		return Function<T, Pow<T>>::apply({base, power});
	}

	static num::Tensor<T> forward(const std::vector<num::Tensor<T>>& args)
	{
		if (args.size() != 2) {
			throw std::invalid_argument("add needs exactly 2 operands");
		}

		return num::applyBinaryWithBroadcast(
			args.at(0), args.at(1),
			[](T a, T b) -> T {
				return std::pow(a, b);
			});
	}

	static void backward(const num::Tensor<T>& outGradient, std::vector<num::Tensor<T>> oldInputs)
	{
		oldInputs[0].setBroadcastGradient(outGradient * forward({oldInputs[0], oldInputs[1] - num::Tensor<T>(1)}) * oldInputs[1]);
	}
};

template <num::num_t T>
inline constexpr Pow<T> pow {};

template <num::num_t T>
class MatMul : public Function<T, MatMul<T>> {
public:
	static num::Tensor<T> operator()(const num::Tensor<T>& a, const num::Tensor<T>& b)
	{
		return Function<T, MatMul<T>>::apply({a, b});
	}

	static num::Tensor<T> forward(const std::vector<num::Tensor<T>>& args)
	{
		if (args.at(0).dims.size() != 2 || args.at(1).dims.size() != 2) {
			throw num::ShapeMismatchError("dot product only defined for 2d arrays");
		}

		num::Tensor<T> out({args.at(0).dims.at(0), args.at(1).dims.at(1)});

		for (int i = 0; i < out.dims.at(0); i++) {
			for (int j = 0; j < out.dims.at(1); j++) {
				T outVal = 0;
				for (int k = 0; k < args.at(0).dims.at(1); k++) {
					outVal += args.at(0).getSingle({i, k}) * args.at(1).getSingle({k, j});
				}
				out.setSingle(outVal, {i, j});
			}
		}
		return out;
	}

	static void backward(const num::Tensor<T>& outGradient, std::vector<num::Tensor<T>> oldInputs)
	{
		// std::cout << "start" << std::endl;
		// std::cout << forward({oldInputs[0].transpose(), outGradient}).toString() << std::endl;
		// std::cout << forward({outGradient, oldInputs[1].transpose()}).toString() << std::endl;
		// std::cout << "end" << std::endl;
		oldInputs[0].setBroadcastGradient(forward({outGradient, oldInputs[1].transpose()}));
		oldInputs[1].setBroadcastGradient(forward({oldInputs[0].transpose(), outGradient}));
	}
};

template <num::num_t T>
inline constexpr MatMul<T> mm {};

template <num::num_t T>
class Transpose : public Function<T, Transpose<T>> {
public:

	static num::Tensor<T> operator()(const num::Tensor<T>& operand)
	{
		return Function<T, Transpose<T>>::apply({operand});
	}

	static num::Tensor<T> forward(const std::vector<num::Tensor<T>>& args)
	{
		if (args.size() != 1) {
			throw std::invalid_argument("transpose needs exactly 2 operands");
		}

		if (args.at(0).dims.size() != 2) {
			throw num::ShapeMismatchError("transpose only defined for 2d arrays");
		}
		num::Tensor<T> out(args.at(0).clone());
		out.dims = num::IntArrRef({args.at(0).dims.at(1), args.at(0).dims.at(0)});
		for (int i = 0; i < args.at(0).dims.at(0); ++i) {
			for (int j = 0; j < args.at(0).dims.at(1); ++j) {
				T val = args.at(0).getSingle({i, j});
				out.setSingle(val, {j, i});
			}
		}
		return out;
	}

	static void backward(const num::Tensor<T>& outGradient, std::vector<num::Tensor<T>> oldInputs)
	{
		oldInputs[0].setBroadcastGradient(forward({outGradient}));
	}
};

template <num::num_t T>
inline constexpr Transpose<T> transpose {};



template <num::num_t T>
class Sigmoid : public Function<T, Sigmoid<T>> {
public:
	static num::Tensor<T> operator()(const num::Tensor<T>& operand)
	{
		return Function<T, Sigmoid<T>>::apply({operand});
	}

	static num::Tensor<T> forward(const std::vector<num::Tensor<T>>& args)
	{
		if (args.size() != 1) {
			throw std::invalid_argument("sigmoid needs exactly one argument");
		}
		num::Tensor<T> one(1);
		return one / (one + (one / args[0].exp()));
	}

	static void backward(const num::Tensor<T>& outGradient, std::vector<num::Tensor<T>> oldInputs)
	{
		num::Tensor<T> out = forward(oldInputs);
		oldInputs[0].setBroadcastGradient(out * outGradient * (num::Tensor<T>(1) - out));
	}
};

template <num::num_t T>
inline constexpr Sigmoid<T> sigmoid {};

template <num::num_t T>
class ReLU : public Function<T, ReLU<T>> {
public:
	static num::Tensor<T> operator()(const num::Tensor<T>& operand)
	{
		return Function<T, ReLU<T>>::apply({operand});
	}

	static num::Tensor<T> forward(const std::vector<num::Tensor<T>>& args)
	{
		if (args.size() != 1) {
			throw std::invalid_argument("ReLU needs exactly one argument");
		}
		return (args[0].clone().applyUnary([](T val) {return (val > 0) ? val : 0;}));
	}

	static void backward(const num::Tensor<T>& outGradient, std::vector<num::Tensor<T>> oldInputs)
	{
		num::Tensor<T> gradient = oldInputs[0].clone().applyUnary([](T val) {return (val > 0) ? 1 : 0;}) * outGradient;
		oldInputs[0].setBroadcastGradient(gradient);
	}
};


template <num::num_t T>
inline constexpr ReLU<T> relu {};
} // namespace autofn

#endif