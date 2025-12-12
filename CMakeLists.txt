cmake_minimum_required(VERSION 3.0.2)
project(similarity_memory)

## Compile as C++ or Python
catkin_python_setup()

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

add_service_files(
  FILES
  ProcessData.srv
  PromptLLM.srv
  MemoryQuery.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)