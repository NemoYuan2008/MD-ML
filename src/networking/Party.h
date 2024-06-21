// By Boshi Yuan

#ifndef MD_ML_PARTY_H
#define MD_ML_PARTY_H


#include <iostream>
#include <string>
#include <vector>
#include <concepts>
#include <cstddef>
#include <mutex>

#include <boost/asio.hpp>


namespace md_ml {


class Party {
public:
    Party(std::size_t p_my_id, std::size_t p_port, std::size_t p_num_parties);

    Party(const Party&) = delete;
    Party& operator=(const Party&) = delete;

    void SendString(std::size_t to_id, const std::string& message);
    std::string ReceiveString(std::size_t from_id);

    void SendInt(std::size_t to_id, int message);
    int ReceiveInt(std::size_t from_id);

    template <std::integral T>
    void Send(std::size_t to_id, T message);

    template <std::integral T>
    T Receive(std::size_t from_id);

    template <std::integral T>
    void SendVec(std::size_t to_id, const std::vector<T>& message);

    template <std::integral T>
    std::vector<T> ReceiveVec(std::size_t from_id, std::size_t num_elements);

    [[nodiscard]] std::size_t my_id() const { return my_id_; }

private:
    [[nodiscard]] boost::asio::ip::port_type WhichPort(std::size_t from_id, std::size_t to_id) const;

    void TryAccept(std::size_t from_id);
    void TryConnect(std::size_t to_id);
    void AcceptHandler(const boost::system::error_code& ec, std::size_t from_id) const;
    void ConnectHandler(const boost::system::error_code& ec, std::size_t to_id);

    inline void CheckID(std::size_t target_id) const;

    static constexpr int kRetryAfterSeconds = 2;

    std::size_t my_id_;
    std::size_t port_base_;
    std::size_t num_parties_;

    boost::asio::io_context io_context_;
    std::vector<boost::asio::ip::tcp::socket> send_sockets_;
    std::vector<boost::asio::ip::tcp::socket> receive_sockets_;
    std::vector<boost::asio::ip::tcp::endpoint> send_endpoints_;
    std::vector<boost::asio::ip::tcp::acceptor> acceptors_;
    std::vector<boost::asio::steady_timer> timers_;
    unsigned long long bytes_sent_ = 0;

    mutable std::mutex cerr_mutex_;
};


inline
void Party::CheckID(std::size_t target_id) const {
    if (target_id >= num_parties_) {
        throw std::invalid_argument("Target ID out of range");
    }
    if (target_id == my_id_) {
        throw std::invalid_argument("Party cannot send to itself");
    }
}


template <std::integral T>
inline void Party::Send(std::size_t to_id, T message) {
    CheckID(to_id);
    boost::asio::write(send_sockets_[to_id], boost::asio::buffer(&message, sizeof(message)));
#ifdef MD_ML_DEBUG_ASIO
    std::lock_guard cerr_lock(cerr_mutex_);
    std::cerr << "Party " << my_id_ << " sent integer " << message << " to party " << to_id << '\n';
#endif
}


template <std::integral T>
T Party::Receive(std::size_t from_id) {
    CheckID(from_id);
    T message;
    boost::asio::read(receive_sockets_[from_id], boost::asio::buffer(&message, sizeof(message)));

#ifdef MD_ML_DEBUG_ASIO
    std::lock_guard cerr_lock(cerr_mutex_);
    std::cerr << "Party " << my_id_ << " received integer " << message << " from party " << from_id << '\n';
#endif

    return message;
}


template <std::integral T>
void Party::SendVec(std::size_t to_id, const std::vector<T>& message) {
    CheckID(to_id);
    boost::asio::write(send_sockets_[to_id], boost::asio::buffer(message));
#ifdef MD_ML_DEBUG_ASIO
    std::lock_guard cerr_lock(cerr_mutex_);
    std::cerr << "Party " << my_id_ << " sent vector to party " << to_id << '\n';
    std::ranges::for_each(message, [](T element) {
        std::cerr << element << ' ';
    });
    std::cerr << '\n';
#endif
}


template <std::integral T>
std::vector<T> Party::ReceiveVec(std::size_t from_id, std::size_t num_elements) {
    CheckID(from_id);
    std::vector<T> message(num_elements);
    boost::asio::read(receive_sockets_[from_id], boost::asio::buffer(message));

#ifdef MD_ML_DEBUG_ASIO
    std::lock_guard cerr_lock(cerr_mutex_);
    std::cerr << "Party " << my_id_ << " received vector from party " << from_id << '\n';
    std::ranges::for_each(message, [](T element) {
        std::cerr << element << ' ';
    });
    std::cerr << '\n';
#endif

    return message;
}


} // namespace md_ml


#endif //MD_ML_PARTY_H
