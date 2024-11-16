#ifndef INT_ARR_REF_H
#define INT_ARR_REF_H

#include <initializer_list>
#include <string>
#include <iostream>
#include <optional>
#include <climits>
#include <memory>

#include "NumErrors.h"

namespace num {
class IntArrRef {
public:
	using iterator = int*;
	using const_iterator = const int*;
	using size_type = size_t;
	using value_type = int;
	using reverse_iterator = std::reverse_iterator<iterator>;

	template <template<typename, typename> typename Container, typename Allocator>
	IntArrRef(Container<int, Allocator> c)
	  : arr {std::make_shared<int[]>(c.size())}, sz {c.size()}
	{
		int idx = 0;
		for (int el : c) {
			arr[idx++] = el;
		}
	}

	IntArrRef(std::initializer_list<int> c)
	  : arr {std::make_shared<int[]>(c.size())}, sz {c.size()}
	{
		int idx = 0;
		for (int el : c) {
			arr[idx++] = el;
		}
	}

	IntArrRef(size_type size, int fill = 0)
	  : arr {std::make_shared<int[]>(size)}, sz {size}
	{
		for (int i = 0; i < size; i++) {
			arr[i] = fill;
		}
	}


	IntArrRef(int *els, size_type size)
	  : arr {new int[size]}, sz {size}
	{
		for (int i = 0; i < size; i++) {
			arr[i] = els[i];
		}
	}


	IntArrRef clone() const noexcept
	{
		return IntArrRef(arr.get(), sz);
	}

	IntArrRef(const IntArrRef& other) = default;

	IntArrRef& operator=(const IntArrRef& other) = default;
	// {
	// 	if (sz != other.sz) {
	// 		delete[] arr;
	// 		sz = other.sz;
	// 		arr = new int[sz];
	// 	}
	// 	for (int i = 0; i < sz; i++) {
	// 		arr[i] = other.arr[i];
	// 	}
	// 	return *this;
	// }

	IntArrRef(IntArrRef&& other) = default;
	//   : arr {other.arr}, sz {other.sz}
	// {
	// 	other.arr = nullptr;
	// }

	~IntArrRef() = default;

	constexpr int& operator[](int idx)
	{
		return arr[idx];
	}

	bool operator==(const IntArrRef& other) const noexcept
	{
		if (other.sz != sz) {
			return false;
		} else {
			for (int i = 0; i < sz; i++) {
				if (arr[i] != other.arr[i]) {
					return false;
				}
			}
			return true;
		}
	}

	friend IntArrRef binaryExpr(const IntArrRef& a, const IntArrRef& b, auto fn)
	{
		if (a.sz != b.sz) {
			throw ShapeMismatchError("can't apply binary expression to unequally sized IntArrRefs");
		}

		IntArrRef out(a.sz);
		for (int i = 0; i < a.sz; ++i) {
			out[i] = fn(a.at(i), b.at(i));
		}
		return out;
	}

	IntArrRef operator+(const IntArrRef& other) const
	{
		return binaryExpr(*this, other, [](int a, int b) {return a + b;});
	}

	IntArrRef operator-(const IntArrRef& other) const
	{
		return binaryExpr(*this, other, [](int a, int b) {return a - b;});
	}

	IntArrRef operator/(const IntArrRef& other) const
	{
		return binaryExpr(*this, other, [](int a, int b) {return a / b;});
	}

	IntArrRef pad(size_type noDims) const noexcept
	{
		IntArrRef out(noDims, 1);
		for (int i = sz - 1, j = noDims - 1;
			i >= 0 && j >= 0; --i, --j) {
			out[j] = arr[i];
		}
		return out;
	}

	bool lessThan(const IntArrRef& other, std::optional<IntArrRef> limitOpt) const
	{
		if (other.sz != sz) {
			throw ShapeMismatchError("can't compare IntArrRefs of unequal size");
		}

		IntArrRef limit = limitOpt.value_or(IntArrRef(sz, INT_MAX));
		// as soon as any entry on the LHS is bigger
		// than the corresponding on the RHS return false
		for (int j = 0; j < sz; j++) {
			if (arr[j] >= limit[j]) {
				return false;
			}
		}
		int i = 0;
		for (; i < sz - 1; ++i) {
			if (arr[i] < other.arr[i]) {
				return true;
			} else if (arr[i] > other.arr[i]) {
				return false;
			}
		}
		return arr[i] < other.arr[i];
	}

	int incr(
		std::optional<IntArrRef> startOpt,
		std::optional<IntArrRef> limitOpt,
		std::optional<IntArrRef> stepOpt)
	{
		IntArrRef start = startOpt.value_or(IntArrRef(sz, 0));
		IntArrRef limit = limitOpt.value_or(IntArrRef(sz, INT_MAX));
		IntArrRef step = stepOpt.value_or(IntArrRef(sz, 1));
		if (limit.sz != sz) {
			throw ShapeMismatchError("limit needs to have same number of dimensions");
		}
		int i = sz - 1;
		for (; i >= 0; --i) {
			arr[i] += step[i];
			if (arr[i] >= limit.arr[i]) {
				arr[i] = start.arr[i];
			} else {
				break;
			}
		}

		return (sz - 2) - i;
	}

	int at(int idx) const
	{
		return arr[idx];
	}
	
	iterator begin() const
	{
		return arr.get();
	}

	iterator end() const
	{
		return arr.get() + sz;
	}

	const_iterator cbegin() const
	{
		return arr.get();
	}

	const_iterator cend() const
	{
		return arr.get() + sz;
	}

	int size() const
	{
		return sz;
	}

	const std::string toString() const
	{
		std::string out ("(");
		for (int i = 0; i < sz; i++) {
			out += std::to_string(arr[i]);
			out += ",";
		}
		out += ")";
		return out;
	}
private:
	std::shared_ptr<int[]> arr;
	size_type sz;
};

} // namespace num
#endif
