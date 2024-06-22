add_executable(test_party_0 test_party_0.cpp)
add_executable(test_party_1 test_party_1.cpp)
add_executable(test_fake_offline test_fake_offline.cpp)

target_link_libraries(test_party_0 ${ONLINE_LIB})
target_link_libraries(test_party_1 ${ONLINE_LIB})
target_link_libraries(test_fake_offline ${FAKE_OFFLINE_LIB})
