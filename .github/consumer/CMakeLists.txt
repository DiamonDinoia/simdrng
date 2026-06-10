cmake_minimum_required(VERSION 3.15)
project(simdrng_consumer LANGUAGES CXX)

# Resolve simdrng (and, when built with xsimd, its find_dependency(xsimd) /
# find_dependency(poet)) purely from the install prefix on CMAKE_PREFIX_PATH.
find_package(simdrng CONFIG REQUIRED)

add_executable(consume consume.cpp)
target_link_libraries(consume PRIVATE simdrng::simdrng)
target_compile_features(consume PRIVATE cxx_std_20)
