cmake_minimum_required(VERSION 3.14)
project(YoloSortTracker)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找必要的依赖库
find_package(OpenCV REQUIRED)
# 寻找OpenVINO开发包
find_package(OpenVINO REQUIRED)

# 包含头文件目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS} # 使用OpenVINO的头文件目录
)

# 添加可执行文件及其源文件
add_executable(tracker
    src/main.cpp
    src/Detector.cpp
    src/Hungarian.cpp
    src/IoU.cpp
    src/Sort.cpp
    src/Track.cpp
)

# 链接依赖库
target_link_libraries(tracker
    ${OpenCV_LIBS}
    openvino::runtime      # 链接OpenVINO运行时库
)
