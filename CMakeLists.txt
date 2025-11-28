cmake_minimum_required(VERSION 3.0.2)
project(similarity_memory)

## Compile as C++ or Python
catkin_python_setup()

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  actionlib_msgs
  actionlib
  message_generation
)

add_service_files(
  FILES
  ProcessData.srv
  LLMQuery.srv
)

add_action_files(
  DIRECTORY action
  FILES
  PromptTextLLM.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  actionlib_msgs
)

catkin_package(
  CATKIN_DEPENDS actionlib_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)