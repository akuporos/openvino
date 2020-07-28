cmake_minimum_required(VERSION 3.3)
project(ovpython)

add_subdirectory(pybind11)
pybind11_add_module(ovpython src/Core.cpp)