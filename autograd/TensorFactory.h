#ifndef NARR_FACTORY_H
#define NARR_FACTORY_H

#include <numeric>
#include <random>
#include <chrono>
#include <concepts>

#include "IntArrRef.h"
#include "Tensor.h"

namespace num {

template <num_t T>
class Tensor;

template <num_t T>
Tensor<T> zeros(const IntArrRef& dims)
{
	return Tensor<T>(dims, [](const IntArrRef& idx) -> T {
		return 0;
	});
}

template <num_t T>
Tensor<T> ones(const IntArrRef& dims)
{
	return Tensor<T>(dims, [](const IntArrRef& idx) -> T {
		return 1;
	});
}

template <num_t T>
Tensor<T> eye(const IntArrRef& dims)
{
	return Tensor<T>(dims, [](const IntArrRef& idx) -> T {
		return static_cast<T>(std::reduce(
			idx.cbegin(), idx.cend(), true,
			[&idx](bool prev, int val) -> bool {
				return prev && (val == idx.at(0));
			}
		));
	});
}
template <num_t T, typename Distribution>
Tensor<T> fromDistribution(const IntArrRef& dims, num_t auto... distParams)
{
	unsigned long t = std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
	std::default_random_engine re;
	re.seed(t);
	Distribution dist(distParams...);
	return Tensor<T>(dims, [&re, &dist](const IntArrRef& idx) -> T {
		return static_cast<T>(dist(re));
	});
}

template <num_t T>
Tensor<T> randn(const IntArrRef& dims, T mean = 0, T stddev = 1)
{
	return fromDistribution<T, std::normal_distribution<>>(dims, mean, stddev);
}

template <num_t T>
Tensor<T> randUniform(const IntArrRef& dims, T min, T max)
{
	if (std::is_floating_point<T>::value) {
		return fromDistribution<T, std::uniform_real_distribution<>>(dims, min, max);
	} else {
		return fromDistribution<T, std::uniform_int_distribution<>>(dims, min, max);
	}
}

} // namespace num

#endif