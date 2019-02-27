PROJECT(lstm-parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-finite-math-only -Wall -Wno-missing-braces -std=c++11 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-funroll-loops -Ofast -march=native")

FIND_PACKAGE(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

if(DEFINED ENV{BOOST_ROOT})
  set(Boost_NO_SYSTEM_PATHS ON)
endif()
set(Boost_REALPATH ON)
find_package(Boost COMPONENTS program_options regex serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

set(LIBS ${LIBS} ${Boost_LIBRARIES})
# set(CMAKE_BUILD_TYPE "Debug")

add_executable(lstm-parser parser.cc)
add_executable(test-corpus test_corpus.cc)
target_link_libraries(lstm-parser ${Boost_LIBRARIES} rt dynet ${LIBS})


add_executable(rnn-mp train_rnnlm-mp.cc)
target_link_libraries(rnn-mp rt dynet ${LIBS})
