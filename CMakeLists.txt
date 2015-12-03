find_program(PYTHON "python3" REQUIRED)
message(STATUS "Found Python: ${PYTHON}")

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(DEPS "${CMAKE_CURRENT_SOURCE_DIR}/atmc/mcopt_wrapper.cpp")
set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build")

set(ATMC_LIBS_LIST mcopt armadillo)
set(ATMC_EXTRA_ARGS_LIST "--std=c++11")
if(${OPENMP_FOUND})
    list(APPEND ATMC_EXTRA_ARGS_LIST ${OpenMP_CXX_FLAGS})
endif()
if(APPLE)
    list(APPEND ATMC_EXTRA_ARGS_LIST "-mmacosx-version-min=${MAC_OSX_VERSION}")
endif()
string(REPLACE ";" "', '" ATMC_LIBS "${ATMC_LIBS_LIST}")
string(REPLACE ";" "', '" ATMC_EXTRA_ARGS "${ATMC_EXTRA_ARGS_LIST}")

configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND ${PYTHON} ${SETUP_PY} build -f
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/timestamp
                   DEPENDS ${DEPS})

add_custom_target(atmc ALL DEPENDS ${OUTPUT})
add_dependencies(atmc mcopt)

install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
