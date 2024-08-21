#ifndef NARR_H
#define NARR_H

#include <initializer_list>
#include <tuple>
#include <string>
#include <vector>
#include <cassert>
#include <functional>
#include <algorithm>
#include <memory>
#include <optional>
#include <cmath>
#include <set>
#include <ranges>
#include <concepts>

namespace num {
template <typename T>
concept num_t = std::convertible_to<T, double>;
}

#include "TensorFactory.h"
#include "IntArrRef.h"
#include "Slice.h"
#include "NumErrors.h"

namespace num {


// n-dimensional array
template <num_t T>
class Tensor {
public:
	using size_type = size_t;
	IntArrRef dims;
	std::function<void(const Tensor<T>&, const std::vector<Tensor<T>>&)> backwardFn;
	std::vector<Tensor<T>> gradGraphChildren;
private:
	std::shared_ptr<T[]> arr;
	std::shared_ptr<T[]> gradArr;
	size_type sz;
public:
	Tensor(const IntArrRef& dimensions, std::function<T(const IntArrRef&)> fillFn = [](const IntArrRef& idx) -> T { return 0; })
	  : dims {dimensions.clone()},
	  	backwardFn ([](const Tensor<T>& grad, const std::vector<Tensor<T>>& oldInputs) {return;}),
		gradGraphChildren ({})
	{
		if (dimensions.size() == 0) {
			throw std::invalid_argument("need at least one dimension");
		}

		sz = 1;
		for (int dim : dims) {
			if (dim < 0) {
				throw std::invalid_argument("can't have negative dimension");
			}
			sz *= dim;
		}
		arr = std::make_shared<T[]>(sz);
		gradArr = std::make_shared<T[]>(sz);

		IntArrRef idx(dims.size());
		for (int dimCarries = 0; dimCarries < dims.size(); dimCarries = idxIncr(idx)) {
			setSingle(fillFn(idx), idx);
		}
	}

	Tensor(T val)
	  : dims ({1}), arr (std::make_shared<T[]>(1)), sz (1),
	    gradArr (std::make_shared<T[]>(1)),
	  	gradGraphChildren ({}),
	  	backwardFn ([](const Tensor<T>& grad, const std::vector<Tensor<T>>& oldInputs) {return;})
	{
		arr[0] = val;
	}

	~Tensor() = default;	

	Tensor(const Tensor<T>& other) = default;

	Tensor<T>& operator=(const Tensor<T>& other) = default;

	Tensor(Tensor<T>&& other) = default;

	Tensor<T>& operator=(Tensor<T>&& other) = default;

	void zeroGradient() noexcept
	{
		for (int i = 0; i < sz; ++i) {
			gradArr[i] = 0;
		}
	}

	Tensor<T> getGradient() const noexcept
	{
		Tensor<T> out(this->clone());
		out.arr = out.gradArr;
		return out;
	}

	void setGradient(const Tensor<T>& grad)
	{
		if (grad.dims != dims) {
			throw ShapeMismatchError("Can't set gradient with array of different dimension\n"
									 "Got gradient of dim " + grad.dims.toString() +
									 "but want dimensions " + dims.toString());
		}

		for (int i = 0; i < sz; ++i) {
			gradArr[i] += grad.arr[i];
		}
	}

	void setBroadcastGradient(const num::Tensor<T>& gradient)
	{
		IntArrRef inputDims = dims.pad(gradient.dims.size());
		if (inputDims.lessThan(gradient.dims, std::nullopt)) {
			Tensor<T> newGrad = zeros<T>(inputDims);
			gradient.iter(
				[&newGrad](const IntArrRef& _, const Tensor<T>& gradIter) {
					newGrad = newGrad + gradIter;
				},
				0);
			setGradient(newGrad.reshape(newGrad.dims.pad(dims.size())));
		} else {
			setGradient(gradient.reshape(dims));
		}
	}

	void backward()
	{
		// use topological sort to create directed graph
		std::vector<Tensor<T>> sorted;
		std::set<std::shared_ptr<T[]>> visited;

		std::function<void(const Tensor<T>&)> topologicalSort;
		topologicalSort =
			[&sorted, &visited, &topologicalSort](const Tensor<T>& arr) -> void {
				if (!(visited.contains(arr.arr))) {
					visited.insert(arr.arr);
					for (const Tensor<T>& child : arr.gradGraphChildren) {
						topologicalSort(child);
					}
					sorted.push_back(arr);
				}
			};
		

		topologicalSort(*this);

		setGradient(ones<T>(dims));

		for (const Tensor<T>& arr : sorted | std::views::reverse) {
			arr.backwardFn(arr.getGradient(), arr.gradGraphChildren);
		}
	}

	// TODO needs to be able to handle negative values as indices
	/// Outputs the Tensor that corresponds to the given slice indeces
	Tensor<T> get(std::initializer_list<IdxSel> idcs) const
	{
		std::vector<std::tuple<int,int,int>> srcRanges;
		std::vector<std::tuple<int,int,int>> dstRanges;
		std::vector<int> outDims;
		int totalSize;
		try {
			std::tie(srcRanges, dstRanges, outDims, totalSize) = Tensor::getCopyRanges(*this, *this, idcs);
		} catch (ShapeMismatchError err) {
			throw ShapeMismatchError{"can't get given slice from array of shape " + dims.toString()};
		}

		Tensor<T> out(outDims);

		copy(*this, out, srcRanges, dstRanges, totalSize);
		return out;
	}


	void set(Tensor<T> val, std::initializer_list<IdxSel> idcs)
	{
		std::vector<std::tuple<int,int,int>> srcRanges;
		std::vector<std::tuple<int,int,int>> dstRanges;
		int totalSize;
		try {
			std::vector<int> ignore;
			std::tie(srcRanges, dstRanges, ignore, totalSize) = Tensor::getCopyRanges(val, *this, idcs);
		} catch (ShapeMismatchError err) {
			throw ShapeMismatchError{"can't set given slice with array of shape " + val.dims.toString()};
		}
		copy(val, *this, dstRanges, srcRanges, totalSize);
	}

	template <typename U>
	friend Tensor<U> applyBinaryWithBroadcast(const Tensor<U>& a, const Tensor<U>& b, auto fn);
	

	Tensor<T> clone() const
	{
		Tensor<T> out(*this);
		out.arr = std::make_shared<T[]>(sz);
		out.gradArr = std::make_shared<T[]>(sz);
		for (int i = 0; i < sz; ++i) {
			out.arr[i] = arr[i];
			out.gradArr[i] = gradArr[i];
		}

		return out;
	}

	Tensor<T> transpose() const
	{
		// only works for 2d Tensors
		if (dims.size() != 2) {
			throw ShapeMismatchError{"can only transpose 2d Tensor but have shape " + dims.toString()};
		}
		Tensor<T> out((*this).clone());
		out.dims = IntArrRef({dims.at(1), dims.at(0)});
		for (int i = 0; i < dims.at(0); i++) {
			for (int j = 0; j < dims.at(1); j++) {
				T val = getSingle({i, j});
				out.setSingle(val, {j, i});
			}
		}
		return out;
	}

	/// returns new Tensor that however will point to same
	/// array contents but with different dims property
	Tensor<T> reshape(IntArrRef newDims) const
	{
		Tensor<T> out(*this);
		if (out.sz != sz) {
			throw ShapeMismatchError{"can't reshape from shape " + dims.toString() + " to " + newDims.toString()};
		}
		out.dims = newDims.clone();
		return out;
	}

	Tensor<T>& applyUnary(auto fn)
	{
		for (int i = 0; i < sz; ++i) {
			arr[i] = fn(arr[i]);
		}
		return *this;
	}
	
	Tensor<T>& pow_(T power)
	{
		return applyUnary([power](T val) { return std::pow(val, power); });
	}

	Tensor<T> pow(T power)
	{
		Tensor<T> out(this->clone());
		return out.pow_(power);
	}


	Tensor<T>& exp_()
	{
		return applyUnary([](T val) { return std::exp(val); });
	}

	Tensor<T> exp() const
	{
		Tensor<T> out(this->clone());
		return out.exp_();
	}

	T getSingle(IntArrRef idx) const
	{
		return arr[getLinIdx(idx)];
	}

	void setSingle(T val, IntArrRef idx)
	{
		arr[getLinIdx(idx)] = val;
	}

	/// does not call function on last step if it doesn't
	/// fully fit into array
	void iter(auto fn, int axis=-1) const
	{
		if (axis > dims.size() || axis > 1) {
			throw ShapeMismatchError("axis outside of dimensions, note that currently only 2d tensors are supported for iter");
		}
		if (axis < 0) {
			axis += dims.size();
		}
		IntArrRef step = IntArrRef(dims.size(), 0);// - IntArrRef(dims.size(), 1);
		step[(axis+1)%2] = 1;
		IntArrRef limit = dims.clone();
		limit[(axis + 1) % 2] = 1;
		Tensor<T> curIter(limit);
		IntArrRef idx(step.size(), 0); // set it to all 0
		IntArrRef offset(step.size(), 0);

		while (idx.lessThan(dims, dims)) {
			curIter.setSingle(getSingle(idx), idx - offset);
			IntArrRef oldIdx = idx.clone();
			if (idx.incr(offset, limit, std::nullopt) >= limit.size() - 1) {
				fn(oldIdx, curIter);
				idx = offset + step;
				offset = offset + step;
				limit = limit + step;
			}
		}
	}

	/// Return a string representing the array
	/// every subarray of a dimension will be enclosed in []
	std::string toString() const
	{
		std::string out = "";

		IntArrRef idx (dims.size(), 0);

		int prevDimsUpdated = dims.size();
		for (int i = 0; i < sz; i++) {
			for (int j = 0; j < prevDimsUpdated; j++) {
				out += "[";
			}
			out += std::to_string(getSingle(idx));
			int dimsUpdated = idxIncr(idx);
			for (int j = 0; j < dimsUpdated; j++) {
				out += "]";
			}
			out += ",";
			prevDimsUpdated = dimsUpdated;
		}
		return out;
	}

private:

	/// increment the index to keep all dimensions correct
	/// RETURNS: the number of carries, i.e. how many dimensions
	/// higher than the last one had to be incremented
	int idxIncr(IntArrRef& idx) const
	{
		int i;
		for (i = idx.size() - 1; i >= 0; i--) {
			idx[i] += 1;
			if (idx[i] >= dims.at(i)) {
				idx[i] = 0;
			} else {
				break;
			}
		}
		return (idx.size() - 1) - i;
	}

	/// Increment index while keeping it in the slice boundaries
	/// specified by ranges.
	/// Each tuple stands for start, end, step in the according dimension
	int idxIncr(IntArrRef& idx, std::vector<std::tuple<int,int,int>> ranges) const
	{
		int i;
		for (i = idx.size() - 1; i >= 0; i--) {
			idx[i] += std::get<2>(ranges[i]);
			if (idx[i] >= std::get<1>(ranges[i])) {
				idx[i] = std::get<0>(ranges[i]);
			} else {
				break;
			}
		}
		return (idx.size() - 2) - i;
	}

	/// @brief compute the ranges for each array in every dimension
	/// across which they can iterate
	/// @tparam T element type of both array operands
	/// @param src copying from this array
	/// @param dst to this array
	/// @param idcs the index selection in the bigger array so that
	/// we copy from equally sized arrays
	/// @return
	/// - srcRanges: the ranges over which src must be iterated
	/// - dstRanges: the ranges over which dst must be iterated
	/// - outDims: the dimensions of the smaller of the arrays
	/// - totalSize: the total number of elements that will be copied
	static std::tuple<
		std::vector<std::tuple<int,int,int>>,
		std::vector<std::tuple<int,int,int>>,
		std::vector<int>,
		int
	>
	getCopyRanges(const Tensor<T>& src, const Tensor<T>& dst, std::initializer_list<IdxSel> idcs)
	{
		std::vector<int> outDims;
		std::vector<std::tuple<int,int,int>> srcRanges;
		std::vector<std::tuple<int,int,int>> dstRanges;

		if (src.dims.size() != dst.dims.size()) {
			throw ShapeMismatchError{
				"shapes " + src.dims.toString() + " and " +
				dst.dims.toString() + " are not compatible"};
		}

		const Tensor<T>& bigger = (src.sz >= dst.sz) ? src : dst;
		const Tensor<T>& smaller = (src.sz >= dst.sz) ? dst : src;

		for (int i = 0; i < bigger.dims.size(); i++) {
			std::tuple<int,int,int> srcRange;
			if (i < idcs.size()) {
				IdxSel idxS = *(idcs.begin() + i);
				if (std::holds_alternative<Slice>(idxS)) {
					srcRange =
						std::get<Slice>(idxS).toRangeTuple(bigger.dims.at(i));
				} else {
					int idx = std::get<int>(idxS);
					srcRange = {idx, idx+1, 1};
				}
			} else {
				srcRange = {0, bigger.dims.at(i), 1};
			}
			int dimSize = Slice::calcDimSize(srcRange);
			outDims.push_back(dimSize);
			srcRanges.push_back(srcRange);
			dstRanges.push_back({0, dimSize, 1});
		}

		return {
			srcRanges,
			dstRanges,
			outDims,
			(src.sz > dst.sz) ? dst.sz : src.sz
		};
	}


	static void copy(const Tensor<T>& src, Tensor<T>& dst,
		const std::vector<std::tuple<int,int,int>>& srcRanges,
		const std::vector<std::tuple<int,int,int>>& dstRanges,
		int totalSize)
	{
		IntArrRef srcIdx (src.dims.size());
		IntArrRef dstIdx (dst.dims.size());
		for (int i = 0; i < srcRanges.size(); i++) {
			srcIdx[i] = std::get<0>(srcRanges[i]);
			dstIdx[i] = std::get<0>(dstRanges[i]);
		}


		for (int i = 0; i < totalSize; i++) {
			T val = src.getSingle(srcIdx);
			dst.setSingle(val, dstIdx);
			src.idxIncr(srcIdx, srcRanges);
			dst.idxIncr(dstIdx, dstRanges);
		}
	}


	int getLinIdx(IntArrRef idx) const
	{
		if (idx.size() != dims.size()) {
			throw std::invalid_argument("index for getSingle needs to have"
				" a value for every dimension");
		}


		int linIdx = 0;
		for (int i = idx.size()-1; i >= 0; --i) {
			// negative indeces mean index from back of array
			int idx_reformatted = (idx[i] >= 0) ? idx[i] : (dims.at(i) + idx[i]);
			if (idx_reformatted >= dims.at(i)) {
				throw IndexError(
					"index out of range "
					+ idx.toString() + " in dimension "
					+ std::to_string(i) + " of array with shape "
					+ dims.toString());
			}
			int add = idx[i];
			for (int j = i + 1; j < dims.size(); j++) {
				add *= dims.at(j);
			}
			linIdx += add;
		}
		return linIdx;
	}


};

template<typename T>
Tensor<T> applyBinaryWithBroadcast(const Tensor<T>& a, const Tensor<T>& b, auto fn)
{
	bool swapped = (a.sz < b.sz || (a.sz == b.sz && b.dims.size() > a.dims.size()));
	const Tensor<T>& bigger = (swapped) ? b : a;
	const Tensor<T>& smaller = (swapped) ? a : b;
	bool broadcastable = (bigger.sz % smaller.sz) == 0;
	int numDimsDiff = (bigger.dims.size() - smaller.dims.size());

	// how many times a dimension has to be repeated to broadcast it
	// initialise with 0 for every dim
	std::vector<int> dimReps(smaller.dims.size(), 0);
	for (int bIdx = bigger.dims.size() - 1; bIdx >= 0; --bIdx) {
		int sIdx = bIdx - numDimsDiff;
		if (sIdx < 0) {
			broadcastable = true;
			dimReps[0] *= bigger.dims.at(bIdx);
		} else {
			if ((bigger.dims.at(bIdx) % smaller.dims.at(sIdx)) != 0) {
				throw ShapeMismatchError{"can't broadcast shapes together: "
					+ a.dims.toString() + " and " + b.dims.toString()};
			}
			dimReps[sIdx] = bigger.dims.at(bIdx) / smaller.dims.at(sIdx);
		}
	}

	IntArrRef bIdx(bigger.dims.size());
	IntArrRef sIdx(smaller.dims.size());

	Tensor<T> out(bigger.clone());
	std::vector<int> dimRepsLeft(dimReps);
	for (int i = 0; i < bigger.sz; i++) {
		T valA = bigger.getSingle(bIdx);
		T valB = smaller.getSingle(sIdx);
		out.setSingle((swapped) ? fn(valB, valA) : fn(valA, valB), bIdx);
		bigger.idxIncr(bIdx);
		int curSIdxDim = smaller.dims.size() - 1;

		while (curSIdxDim >= 0 && ++(sIdx[curSIdxDim]) >= smaller.dims.at(curSIdxDim)) {
			sIdx[curSIdxDim] = 0;
			if (--(dimRepsLeft[curSIdxDim]) <= 0) {
				dimRepsLeft[curSIdxDim] = dimReps[curSIdxDim];
				curSIdxDim--;
			} else {
				break;
			}
		}
	}
	return out;
}

} // namespace num

#endif