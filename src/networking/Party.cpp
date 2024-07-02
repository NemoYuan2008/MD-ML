// By Boshi Yuan

#include "Party.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstddef>
#include <mutex>

#include <boost/asio.hpp>

#include "utils/uint128_io.h"


namespace md_ml {

Party::Party(std::size_t p_my_id, std::size_t p_num_parties, std::size_t p_port)
    : my_id_(p_my_id), num_parties_(p_num_parties), port_base_(p_port) {
    send_sockets_.reserve(num_parties_);
    send_endpoints_.reserve(num_parties_);
    receive_sockets_.reserve(num_parties_);
    timers_.reserve(num_parties_);
    acceptors_.reserve(num_parties_);

    for (std::size_t other_id = 0; other_id < num_parties_; ++other_id) {
        // We don't skip my_id_ when constructing the vectors, to avoid confusion on the indices
        send_sockets_.emplace_back(io_context_);
        receive_sockets_.emplace_back(io_context_);
        timers_.emplace_back(io_context_);

        send_endpoints_.emplace_back(boost::asio::ip::make_address("127.0.0.1"), WhichPort(my_id_, other_id));

        // receive_endpoint is used only once and is not stored
        boost::asio::ip::tcp::endpoint receive_endpoint(boost::asio::ip::tcp::v4(), WhichPort(other_id, my_id_));
        acceptors_.emplace_back(io_context_, receive_endpoint);
    }

    // start listening from other parties
    for (std::size_t from_id = 0; from_id < num_parties_; ++from_id) {
        if (from_id == my_id_) {
            continue;
        }
        TryAccept(from_id);
    }

    // connect to other parties
    for (std::size_t to_id = 0; to_id < num_parties_; ++to_id) {
        if (to_id == my_id_) {
            continue;
        }
        TryConnect(to_id);
    }

    io_context_.run();

    // These can be safely deleted
    send_endpoints_.clear();
    send_endpoints_.shrink_to_fit();
    acceptors_.clear();
    acceptors_.shrink_to_fit();
    timers_.clear();
    timers_.shrink_to_fit();
}


boost::asio::ip::port_type Party::WhichPort(std::size_t from_id, std::size_t to_id) const {
    std::size_t ret = port_base_ + from_id * num_parties_ + to_id;
    if (ret > 65535) {
        throw std::invalid_argument("Port number exceeds 65535");
    }
    return static_cast<boost::asio::ip::port_type>(ret);
}


void Party::TryAccept(std::size_t from_id) {
    acceptors_[from_id].async_accept(
        receive_sockets_[from_id],
        [this, from_id](const boost::system::error_code& ec) {
            AcceptHandler(ec, from_id);
        }
    );
}


void Party::TryConnect(std::size_t to_id) {
    send_sockets_[to_id].async_connect(
        send_endpoints_[to_id],
        [to_id, this](const boost::system::error_code& ec) {
            this->ConnectHandler(ec, to_id);
        }
    );
}


void Party::AcceptHandler(const boost::system::error_code& ec, std::size_t from_id) const {
#ifdef MD_ML_DEBUG_ASIO
    std::lock_guard cerr_lock(cerr_mutex_);
    if (!ec) {
        std::cerr << "Party " << my_id_ << " accepted party " << from_id << '\n';
    }
    else {
        std::cerr << "Accept failed: " << ec.message() << '\n';
    }
#endif
}


void Party::ConnectHandler(const boost::system::error_code& ec, std::size_t to_id) {
    if (!ec) {
        // Connection successful
        return;
    }

    std::lock_guard cerr_lock(cerr_mutex_);
    // std::cerr << std::format("Failed to connect to party {}, retry after {} seconds...\n", to_id, kRetryAfterSeconds);
    std::cerr << "Failed to connect to party " << to_id << ", retry after " << kRetryAfterSeconds << " seconds...\n";

    if (send_sockets_[to_id].is_open()) {
        send_sockets_[to_id].close();
    }

    timers_[to_id].expires_from_now(boost::asio::chrono::seconds(kRetryAfterSeconds));
    timers_[to_id].async_wait([to_id, this](const boost::system::error_code&) {
        this->TryConnect(to_id);
    });
}


void Party::SendString(std::size_t to_id, const std::string& message) {
    CheckID(to_id);

    // We don't handle the exception from boost.asio here, we are happy to let the program terminate anyway
    boost::asio::write(send_sockets_[to_id], boost::asio::buffer(message));

#ifdef MD_ML_DEBUG_ASIO
    std::lock_guard lock(cerr_mutex_);
    std::cerr << "Party " << my_id_ << " sent: " << message << " to party " << to_id << '\n';
#endif
}


std::string Party::ReceiveString(std::size_t from_id) {
    CheckID(from_id);

    std::vector<char> buffer(1024);
    size_t length = receive_sockets_[from_id].read_some(boost::asio::buffer(buffer));
    std::string message(buffer.data(), length);

#ifdef MD_ML_DEBUG_ASIO
    std::lock_guard lock(cerr_mutex_);
    std::cerr << "Party " << my_id_ << " received: " << message << " from party " << from_id << '\n';
#endif

    return message;
}


// Send an integer
void Party::SendInt(std::size_t to_id, int message) {
    CheckID(to_id);
    boost::asio::write(send_sockets_[to_id], boost::asio::buffer(&message, sizeof(message)));

#ifdef MD_ML_DEBUG_ASIO
    std::lock_guard lock(cerr_mutex_);
    std::cerr << "Party " << my_id_ << " sent integer " << message << " to party " << to_id << '\n';
#endif
}


// Receive an integer
int Party::ReceiveInt(std::size_t from_id) {
    CheckID(from_id);

    int message;
    boost::asio::read(receive_sockets_[from_id], boost::asio::buffer(&message, sizeof(message)));

#ifdef MD_ML_DEBUG_ASIO
    std::lock_guard lock(cerr_mutex_);
    std::cerr << "Party " << my_id_ << " received integer " << message << " from party " << from_id << '\n';
#endif

    return message;
}


// explicit instantiate the template functions
// template void Party::Send<uint64_t>(std::size_t, uint64_t);
// template void Party::Send<__uint128_t>(std::size_t, __uint128_t);
// template uint64_t Party::Receive<uint64_t>(std::size_t);
// template __uint128_t Party::Receive<__uint128_t>(std::size_t);
// template void Party::SendVec<uint64_t>(std::size_t, const std::vector<uint64_t>&);
// template void Party::SendVec<__uint128_t>(std::size_t, const std::vector<__uint128_t>&);
// template std::vector<uint64_t> Party::ReceiveVec<uint64_t>(std::size_t, std::size_t);
// template std::vector<__uint128_t> Party::ReceiveVec<__uint128_t>(std::size_t, std::size_t);

} // namespace md_ml
