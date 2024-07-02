add_executable(AlexNet_party0 AlexNetParty0.cpp AlexNetConfig.h)
add_executable(AlexNet_party1 AlexNetParty1.cpp AlexNetConfig.h)
add_executable(AlexNet_fake_offline AlexNetFakeOffline.cpp AlexNetConfig.h)

target_link_libraries(AlexNet_party0 ${ONLINE_LIB})
target_link_libraries(AlexNet_party1 ${ONLINE_LIB})
target_link_libraries(AlexNet_fake_offline ${FAKE_OFFLINE_LIB})