set(SRC_NETWORKING
        Party.cpp
        Party.h
)

add_library(md-ml-networking ${SRC_NETWORKING})

# For Windows, we need to link against ws2_32 and wsock32 for sockets
if (WIN32)
    target_link_libraries(md-ml-networking ws2_32 wsock32)
endif ()
