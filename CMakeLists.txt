cmake_minimum_required(VERSION 3.28)
project ("CppAutoGrad")

add_compile_options(-std=c++23)

include(FetchContent) # If not included already

FetchContent_Declare(sciplot_content
	GIT_REPOSITORY https://github.com/sciplot/sciplot.git
	GIT_TAG master)

FetchContent_GetProperties(sciplot_content)
if(NOT sciplot_content_POPULATED)
	FetchContent_Populate(sciplot_content)
endif()

include_directories(${sciplot_content_SOURCE_DIR})

add_executable("model_demo" "model_demo.cpp")
add_executable("grad_demo" "grad_demo.cpp")

target_include_directories("model_demo" PUBLIC "${sciplot_content_SOURCE_DIR}")