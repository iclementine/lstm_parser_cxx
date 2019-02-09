PROJECT(lstm-parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11")

if(DEFINED ENV{BOOST_ROOT})
  set(Boost_NO_SYSTEM_PATHS ON)
endif()
set(Boost_REALPATH ON)
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
set(LIBS ${LIBS} ${Boost_LIBRARIES})
# set(CMAKE_BUILD_TYPE "Debug")
add_executable(lstm-parser parser.cc)
add_executable(test-corpus test_corpus.cc)
target_link_libraries(lstm-parser ${Boost_LIBRARIES} rt dynet)