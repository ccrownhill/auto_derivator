#ifndef NARR_OPS_H
#define NARR_OPS_H

#include "Tensor.h"
#include "AutogradFunction.h"

namespace num {

template <num_t T>
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b)
{
	return autofn::Add<T>::apply({a, b});
}

template <num_t T>
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b)
{
	return autofn::Sub<T>::apply({a, b});
}

template <num_t T>
Tensor<T> operator-(const Tensor<T>& a)
{
	return autofn::Sub<T>::apply({zeros<T>(a.dims), a});
}


template <num_t T>
Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b)
{
	return autofn::Mul<T>::apply({a, b});
}

template <num_t T>
Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b)
{
	return autofn::Div<T>::apply({a, b});
}

/// compute dot product of Tensors a and b
template <num_t T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b)
{
	return autofn::MatMul<T>::apply({a, b});
}
} // namespace num

#endif