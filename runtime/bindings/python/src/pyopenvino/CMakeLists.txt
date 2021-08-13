# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

project (pyopenvino)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11 EXCLUDE_FROM_ALL)

# PYTHON_VERSION_MAJOR and PYTHON_VERSION_MINOR are defined inside pybind11
set(PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
message(STATUS "Python version=${PYTHON_VERSION}")

set(LIBRARY_OUTPUT_DIRECTORY_BIN ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
if(OpenVINO_SOURCE_DIR)
    if(WIN32)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/${PYTHON_VERSION}/openvino)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/${PYTHON_VERSION}/openvino)
    endif()

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
endif()

# compile options

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # disable warning: This operator was deprecated and will be removed with v0 operation.
    add_compile_options(/wd4996)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-deprecated-register -Wno-range-loop-analysis)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_link_options(-stdlib=libc++)
    add_compile_options(-Wno-unused-value -Wno-range-loop-analysis)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # WA for GCC 7.5 "PYBIND11_NOINLINE inline" warning
    add_compile_options(-Wno-error=attributes)
endif()

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # for proper fix need to update pybind to version which does not use PyEval_InitThreads()
    add_compile_options(-Wno-deprecated-declarations -Wno-undef)
endif()

if(DEFINED IE_MAIN_SOURCE_DIR)
    set(InferenceEngine_LIBRARIES inference_engine)
    set(NGRAPH_LIBRARIES ngraph)
else()
    find_package(InferenceEngineDeveloperPackage REQUIRED)
    find_package(ngraph REQUIRED)
    set(InferenceEngine_LIBRARIES IE::inference_engine)
    set(NGRAPH_LIBRARIES ngraph)
endif()

# create target
file(GLOB_RECURSE SOURCES core/*.cpp graph/*.cpp)

pybind11_add_module(${PROJECT_NAME} MODULE ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
# ${NGRAPH_LIBRARIES}
target_link_libraries(${PROJECT_NAME} PRIVATE ${InferenceEngine_LIBRARIES} ngraph::ngraph ngraph::frontend_manager)

if (TARGET ngraph::onnx_importer)
    add_dependencies(${PROJECT_NAME} ngraph::onnx_importer)
endif()

if(NGRAPH_UNIT_TEST_ENABLE)
    add_subdirectory(tests/mock/mock_py_ngraph_frontend)
    add_dependencies(${PROJECT_NAME} mock_py_ngraph_frontend)
    set_target_properties(mock_py_ngraph_frontend PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            COMPILE_PDB_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            PDB_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN})

    add_subdirectory(tests/mock/pyngraph_fe_mock_api)
    add_dependencies(${PROJECT_NAME} pybind_mock_frontend)
endif()

# perform copy
if(OpenVINO_SOURCE_DIR)
    add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/../openvino ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            # COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${PROJECT_NAME}.*.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            )
endif()

message(STATUS "diiiiiiiir" ${CMAKE_CURRENT_LIST_DIR})

if(OpenVINO_SOURCE_DIR OR InferenceEngineDeveloperPackage_FOUND)
    if(COMMAND ie_python_minimal_api)
        ie_python_minimal_api(${PROJECT_NAME})
    endif()

    add_clang_format_target(${PROJECT_NAME}_clang FOR_TARGETS ${PROJECT_NAME})

    ie_cpack_add_component(pyopenvino_${PYTHON_VERSION})

    install(TARGETS ${PROJECT_NAME}
            DESTINATION python/${PYTHON_VERSION}/openvino
            COMPONENT pyopenvino_${PYTHON_VERSION})

    install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../openvino
            DESTINATION python/${PYTHON_VERSION}
            COMPONENT pyopenvino_${PYTHON_VERSION}
            USE_SOURCE_PERMISSIONS)

    install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../tests
            DESTINATION tests/${PROJECT_NAME}
            COMPONENT tests EXCLUDE_FROM_ALL)

    ie_cpack(pyopenvino_${PYTHON_VERSION})
endif()
