#ifndef NUM_ERRORS_H
#define NUM_ERRORS_H

#include <stdexcept>

namespace num {
class IndexError : public std::runtime_error {
public:
	IndexError(const std::string& msg)
		: std::runtime_error("IndexError: " + msg) {}
};

class ShapeMismatchError : public std::runtime_error {
public:
	ShapeMismatchError(const std::string& msg)
		: std::runtime_error("ShapeMismatchError: " + msg) {}
};
} // namespace num

#endif