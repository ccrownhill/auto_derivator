#ifndef SLICE_H
#define SLICE_H

#include <optional>
#include <tuple>
#include <variant>


namespace num {

struct Slice {
	std::optional<int> start;
	std::optional<int> end;
	std::optional<int> step;

	std::tuple<int, int, int> toRangeTuple(int dimSize) const
	{
		int startVal = start.value_or(0);
		int endVal = end.value_or(dimSize);
		int stepVal = step.value_or(1);
		return {startVal, endVal, stepVal};
	}

	static int calcDimSize(std::tuple<int, int, int> range)
	{
		auto [start, end, step] = range;
		return Slice::calcDimSize(start, end, step);
	}

	static int calcDimSize(int start, int end, int step)
	{
		return (end - start  + (step - 1)) / step;
	}
};

using IdxSel = std::variant<Slice, int>;

} // namespace num

#endif