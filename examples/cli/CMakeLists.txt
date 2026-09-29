add_executable(parakeet-cli main.cpp)
target_link_libraries(parakeet-cli PRIVATE parakeet)
target_include_directories(parakeet-cli PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(diarize diarize.cpp)
target_link_libraries(diarize PRIVATE parakeet)
target_include_directories(diarize PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

add_executable(sound-window-eval sound_window_eval.cpp)
target_link_libraries(sound-window-eval PRIVATE parakeet)
target_include_directories(sound-window-eval PRIVATE ${CMAKE_SOURCE_DIR}/src)
