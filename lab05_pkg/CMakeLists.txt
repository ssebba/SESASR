cmake_minimum_required(VERSION 3.8)
project(lab05_pkg)

# --- 1. Trova i pacchetti necessari per compilare l'Action ---
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)

# --- 2. Genera l'Action ---
# Assicurati che il percorso "action/FollowTarget.action" sia corretto!
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/FollowTarget.action"
  DEPENDENCIES action_msgs
)

# --- 3. Chiusura standard ---
ament_package()