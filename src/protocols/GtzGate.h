// By Shixuan Yang

#ifndef GTZGATE_H
#define GTZGATE_H

#include <memory>
#include <vector>
#include <bitset>
#include <thread>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"


namespace md_ml {

template <IsSpdz2kShare ShrType>
class GtzGate : public Gate<ShrType> {
public:
    using ClearType = typename ShrType::ClearType;
    using SemiShrType = typename ShrType::SemiShrType;

    explicit GtzGate(const std::shared_ptr<Gate<ShrType>>& p_input_x);

private:
    void doReadOfflineFromFile() override;

    void doRunOnline() override;

    // template <typename T>
    // bool CarryOutAux(std::bitset<sizeof(T) * 8> p,
    //                  std::bitset<sizeof(T) * 8> g, int k);

    std::vector<bool> BitLT(std::vector<ClearType>& pInt, // output s = (pInt < sInt)
                            std::vector<ClearType>& sInt);

    std::vector<bool> CarryOutCin(std::vector<std::bitset<sizeof(ClearType) * 8>>& aIn,
                                  std::vector<std::bitset<sizeof(ClearType) * 8>>& bIn,
                                  bool cIn); // a<-delta_x, b<-lambda_xBinShr

    std::vector<bool> CarryOutAux(std::vector<std::bitset<sizeof(ClearType) * 8>> p,
                                  std::vector<std::bitset<sizeof(ClearType) * 8>> g,
                                  int k); // k bits, (p2,g2)*(p1,g1) = (p2p1,g2+p2g1)

    std::vector<ClearType> lambda_xBinShr;

    // Binary Triples, we currently fake them as (0, 0, 0)
    bool a = false;
    bool b = false;
    bool c = false;
};


template <IsSpdz2kShare ShrType>
GtzGate<ShrType>::
GtzGate(const std::shared_ptr<Gate<ShrType>>& p_input_x)
    : Gate<ShrType>(p_input_x, nullptr) {
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_x->dim_col());
}

template <IsSpdz2kShare ShrType>
void GtzGate<ShrType>::doReadOfflineFromFile() {
    auto size = this->dim_row() * this->dim_col();

    this->lambda_shr() = this->party().ReadShares(size);
    this->lambda_shr_mac() = this->party().ReadShares(size);
    lambda_xBinShr = this->party().ReadClear(size);
}

template <IsSpdz2kShare ShrType>
void GtzGate<ShrType>::doRunOnline() {
#ifndef NDEBUG
    //        std::cout << "\nGtzGate Online\n";
    //        std::cout << "lambdaShr:";
    //        printVector(this->lambdaShr);
    //        std::cout << "deltaClear:";
    //        printVector(this->deltaClear);
    //        std::cout << "lambda_xBinShr:";
    //        printVector(this->lambda_xBinShr);
#endif

    // const auto& delta_x_semiShr = this->input_x->getDeltaClear();
    const auto& delta_x_semiShr = this->input_x()->Delta_clear(); // By ybs

    std::vector<ClearType> delta_x(delta_x_semiShr.begin(), delta_x_semiShr.end());
    auto ret = BitLT(delta_x, this->lambda_xBinShr);
    if (this->my_id() == 0) {
        std::transform(ret.begin(), ret.end(), ret.begin(), std::logical_not<>());
    }

#ifndef NDEBUG
    //        std::cout << "delta_x: ";
    //        printVector(delta_x);
    //        std::cout << "Lambda_xBinShr: ";
    //        printVector(this->lambda_xBinShr);
    //        std::cout << "Ret value of BitLT:";
    //        printVector(ret);
#endif

    //TODO: this is fake

    int msgBytes = (ret.size() + 7) / 8; // round up
    //        std::vector<std::bitset<sizeof(uint8_t) * 8>> sendmsg(msgBytes, 0);
    //        std::vector<std::bitset<sizeof(uint8_t) * 8>> rcvmsg(msgBytes, 0);

    std::vector<uint8_t> sendmsg(msgBytes, 0);
    // std::vector<uint8_t> rcvmsg(msgBytes, 0);
    std::vector<uint8_t> rcvmsg; // By ybs
    int vec_loc, bit_loc;
    for (int j = 0; j < ret.size(); ++j) {
        vec_loc = j / 8;
        bit_loc = j % 8;
        sendmsg[vec_loc] += ret[j] << bit_loc; //[p1] -[a]
    }
#ifndef NDEBUG
    std::cout << "GtzGate open ret value, size: " << sendmsg.size() << "\n";
#endif
    // std::thread t1([this, &sendmsg]() { this->getParty()->getNetwork().send(1 - this->myId(), sendmsg); });
    // std::thread t2(
    //     [this, &rcvmsg] { this->getParty()->getNetwork().rcv(1 - this->myId(), &rcvmsg, rcvmsg.size()); });
    std::thread t1([this, &sendmsg] { this->party().SendVecToOther(sendmsg); }); // By ybs
    std::thread t2(
        [this, &rcvmsg, msgBytes] {
            rcvmsg = this->party().template ReceiveVecFromOther<uint8_t>(msgBytes);
        }); // By ybs
    t1.join();
    t2.join();

    std::vector<bool> rcv(ret.size());
    for (int j = 0; j < rcv.size(); ++j) {
        vec_loc = j / 8;
        bit_loc = j % 8;
        rcv[j] = std::bitset<sizeof(uint8_t) * 8>(rcvmsg[vec_loc])[bit_loc]; //[p1] -[a]
    }


    std::vector<SemiShrType> zShr(ret.size(), 0);

    if (this->my_id() == 0) {
        std::transform(ret.begin(), ret.end(), rcv.begin(), zShr.begin(), std::bit_xor<>());
    }

    // this->deltaClear = matrixAdd(this->lambdaShr, zShr);
    this->Delta_clear() = matrixAdd(this->lambda_shr(), zShr); // By ybs
    std::vector<SemiShrType> deltaRcv(zShr.size());

#ifndef NDEBUG
    // std::cout << "GtzGate open deltaClear, size: " << this->deltaClear.size() << "\n";
    std::cout << "GtzGate open deltaClear, size: " << this->Delta_clear().size() << "\n";
#endif
    // std::thread t3([this]() { this->getParty()->getNetwork().send(1 - this->myId(), this->deltaClear); });
    // std::thread t4([this, &deltaRcv]() {
    //     this->getParty()->getNetwork().rcv(1 - this->myId(), &deltaRcv, deltaRcv.size());
    // });

    // By ybs begin
    auto size = zShr.size();
    std::thread t3([this] { this->party().SendVecToOther(this->Delta_clear()); });
    std::thread t4(
        [this, &deltaRcv, size] {
            deltaRcv = this->party().template ReceiveVecFromOther<SemiShrType>(size);
        });
    // By ybs end

    t3.join();
    t4.join();

    // matrixAddAssign(this->deltaClear, deltaRcv);
    matrixAddAssign(this->Delta_clear(), deltaRcv); // By ybs
}


// template <IsSpdz2kShare ShrType>
// template <typename T>
// bool GtzGate<ShrType>::
// CarryOutAux(std::bitset<sizeof(T) * 8> p, std::bitset<sizeof(T) * 8> g, int k) {
//     //k bits, (p2,g2)*(p1,g1) = (p2p1,g2+p2g1)
//     if (k > 1) {
//         int u_len = k / 2; // round down bit length, if k%2=1, push back the last one bit at the end
//         std::bitset<sizeof(T) * 8> u_p, u_g;
//         // compute u[k/2..1] = (d[2k] * d[2k-1],...)---- compute p2*p1 and g2*p1 need 2 triples
//         // (k/2)*2 triples per invocation
//         int numTriples = u_len * 2; //parallel g.size() comparisons
//         //prepare beaver's triples
//         //[alpha] = [x] - [a]
//         //[beta] = [y] - [b]
//         // open alpha, beta
//         // compute [z] = [c] + alpha*[b] + beta*[a] + alpha*beta
//
//         // each triple sholud send alpha, beta -- 2 bits
//         int msgBytes = (numTriples * 2 + 7) / 8; // round up
//         //            std::vector<std::bitset<sizeof(uint8_t)*8>> sendmsg(msgBytes,0);
//         //            std::vector<std::bitset<sizeof(uint8_t)*8>> rcvmsg(msgBytes,0);
//
//         std::vector<uint8_t> sendmsg(msgBytes, 0);
//         std::vector<uint8_t> rcvmsg(msgBytes, 0);
//         int vec_loc, bit_loc;
//         int index_triple = 0;
//         // load the msg
//         for (int j = 0; j < u_len; ++j) {
//             vec_loc = index_triple * 2 / 8;
//             bit_loc = (index_triple * 2) % 8;
//             index_triple++;
//             sendmsg[vec_loc] += (p[2 * j] ^ a) << bit_loc; //[p1] -[a]
//             sendmsg[vec_loc] += (p[2 * j + 1] ^ b) << (bit_loc + 1); //[p2] -[b]
//             vec_loc = index_triple * 2 / 8;
//             bit_loc = (index_triple * 2) % 8;
//             index_triple++;
//             sendmsg[vec_loc] = (g[2 * j] ^ a) << bit_loc; //[g1] -[a]
//             sendmsg[vec_loc] = (p[2 * j + 1] ^ b) << (bit_loc + 1); //[p2] -[b]
//         }
//         //send & rcv
//         //send numTriples, sendmsg; receive numTriples rcvmsg
//         std::thread t1([this, &sendmsg]() {
//             this->party->getNetwork().send(1 - this->myId(), sendmsg);
//         });
//         std::thread t2([this, &rcvmsg]() {
//             this->party->getNetwork().rcv(1 - this->myId(), &rcvmsg, rcvmsg.size());
//         });
//         t1.join();
//         t2.join();
//         //#ifndef NDEBUG
//         //            std::cout<<"sendmsg: ";printVector(sendmsg);
//         //            std::cout<<"rcvmsg: "; printVector(rcvmsg);
//         //#endif
//         //compute
//         bool alpha, beta, z, x_, y_;
//         index_triple = 0;
//         for (int j = 0; j < u_len; ++j) {
//             // compute u_p, u_g
//             vec_loc = index_triple * 2 / 8;
//             bit_loc = (index_triple * 2) % 8;
//             index_triple++;
//             alpha = std::bitset<sizeof(uint8_t) * 8>(sendmsg[vec_loc])[bit_loc] ^ std::bitset<sizeof(uint8_t) *
//                 8>(rcvmsg[vec_loc])[bit_loc]; // alpha = p1 -a
//             beta = std::bitset<sizeof(uint8_t) * 8>(sendmsg[vec_loc])[bit_loc + 1] ^ std::bitset<sizeof(uint8_t) *
//                 8>(rcvmsg[vec_loc])[bit_loc + 1]; // beta = p2 - b
//             x_ = p[2 * j], y_ = p[2 * j + 1]; //x_ -- p1, y_ -- p2
//             z = c ^ (alpha & y_) ^ (beta & x_); // z = p1p2
//             if (this->myId() == 0) z ^= (alpha & beta);
//             u_p[j] = z;
//             vec_loc = index_triple * 2 / 8;
//             bit_loc = (index_triple * 2) % 8;
//             index_triple++;
//             alpha = std::bitset<sizeof(uint8_t) * 8>(sendmsg[vec_loc])[bit_loc] ^ std::bitset<sizeof(uint8_t) *
//                 8>(rcvmsg[vec_loc])[bit_loc]; // open g1 -a
//             beta = std::bitset<sizeof(uint8_t) * 8>(sendmsg[vec_loc])[bit_loc + 1] ^ std::bitset<sizeof(uint8_t) *
//                 8>(rcvmsg[vec_loc])[bit_loc + 1]; // open  p2 -b
//             x_ = g[2 * j], y_ = p[2 * j + 1]; // x_ -- g1 y_ -- p2
//             z = c ^ (alpha & y_) ^ (beta & x_); // z = p2g1
//             if (this->myId() == 0) z ^= (alpha & beta);
//             u_g[j] = g[2 * j + 1] ^ z; // u_g = g2 + p2g1
//         }
//         if (index_triple < numTriples) std::cout << "triples amount error\n";
//         if (k % 2 == 1) {
//             u_len += 1;
//             for (int i = 0; i < p.size(); ++i) {
//                 u_p[i][u_len] = p[i][k - 1];
//                 u_g[i][u_len] = g[i][k - 1];
//             }
//         }
//         auto ret = CarryOutAux(u_p, u_g, u_len); // u_len : bit length
//         return ret;
//     }
//     else {
//         return g[0]; // Actcually only care g[..][0]
//     }
// }


template <IsSpdz2kShare ShrType>
std::vector<bool> GtzGate<ShrType>::
BitLT(std::vector<ClearType>& pInt, std::vector<ClearType>& sInt) {
    // output s = (pInt < sInt)
    std::vector<std::bitset<sizeof(ClearType) * 8>> b_(sInt.size());
    std::vector<std::bitset<sizeof(ClearType) * 8>> a_(pInt.size());
    for (int i = 0; i < sInt.size(); ++i) {
        if (this->my_id() == 0) b_[i] = ~sInt[i]; //b_[i][j] = 1 - b[i][j]
        else b_[i] = sInt[i];
        a_[i] = pInt[i];
    }
    auto s = CarryOutCin(a_, b_, 1);
    for (int i = 0; i < s.size(); ++i) {
        if (this->my_id() == 0) s[i] = 1 ^ s[i]; //s[i] = 1 -s[i]
    }
    return s;
}


template <IsSpdz2kShare ShrType>
std::vector<bool> GtzGate<ShrType>::
CarryOutCin(std::vector<std::bitset<sizeof(ClearType) * 8>>& aIn,
            std::vector<std::bitset<sizeof(ClearType) * 8>>& bIn,
            bool cIn) {
    //a<-delta_x, b<-lambda_xBinShr
    std::vector<std::bitset<sizeof(ClearType) * 8>> p(bIn.size());
    std::vector<std::bitset<sizeof(ClearType) * 8>> g(bIn.size());
    //compute p[i] = a[i]^b[i], g[i] = a[i]*b[i]
    int numBits = sizeof(ClearType) * 8;
    for (int i = 0; i < bIn.size(); ++i) {
        for (int j = 0; j < numBits; j++) {
            if (this->my_id() == 0) p[i][j] = aIn[i][j] ^ bIn[i][j];
            else p[i][j] = bIn[i][j]; //p = a+b -2ab
            g[i][j] = aIn[i][j] & bIn[i][j]; //g = a*b
        }
    }
    for (int i = 0; i < bIn.size(); ++i) {
        g[i][0] = g[i][0] ^ (cIn & p[i][0]); // g1 = g1 + c*p1
    }
    return CarryOutAux(p, g, numBits);
}

template <IsSpdz2kShare ShrType>
std::vector<bool> GtzGate<ShrType>::
CarryOutAux(std::vector<std::bitset<sizeof(ClearType) * 8>> p,
            std::vector<std::bitset<sizeof(ClearType) * 8>> g,
            int k) {
    //k bits, (p2,g2)*(p1,g1) = (p2p1,g2+p2g1)
    if (k > 1) {
        int u_len = k / 2; // round down bit length, if k%2=1, push back the last one bit at the end
        std::vector<std::bitset<sizeof(ClearType) * 8>> u_p(p.size(), 0);
        std::vector<std::bitset<sizeof(ClearType) * 8>> u_g(p.size(), 0);
        // compute u[k/2..1] = (d[2k] * d[2k-1],...)---- compute p2*p1 and g2*p1 need 2 triples
        // (k/2)*2 triples per invocation
        int numTriples = g.size() * u_len * 2; //parallel g.size() comparisons
        //prepare beaver's triples
        //[alpha] = [x] - [a]
        //[beta] = [y] - [b]
        // open alpha, beta
        // compute [z] = [c] + alpha*[b] + beta*[a] + alpha*beta

        // each triple should send alpha, beta -- 2 bits
        int msgBytes = (numTriples * 2 + 7) / 8; // round up
        std::vector<uint8_t> sendmsg(msgBytes, 0);
        std::vector<uint8_t> rcvmsg(msgBytes, 0);
        std::vector<ClearType> sendMacmsg(numTriples * 2, 0);
        std::vector<ClearType> rcvMacmsg(numTriples * 2, 0);
        int vec_loc, bit_loc;
        int index_triple = 0;
        // load the msg
        for (int i = 0; i < g.size(); ++i) {
            // parallel comparison
            for (int j = 0; j < u_len; ++j) {
                vec_loc = index_triple * 2 / 8;
                bit_loc = (index_triple * 2) % 8;
                index_triple++;
                sendmsg[vec_loc] += (p[i][2 * j] ^ a) << bit_loc; //[p1] -[a]
                sendmsg[vec_loc] += (p[i][2 * j + 1] ^ b) << (bit_loc + 1); //[p2] -[b]
                vec_loc = index_triple * 2 / 8;
                bit_loc = (index_triple * 2) % 8;
                index_triple++;
                sendmsg[vec_loc] += (g[i][2 * j] ^ a) << bit_loc; //[g1] -[a]
                sendmsg[vec_loc] += (p[i][2 * j + 1] ^ b) << (bit_loc + 1); //[p2] -[b]
            }
        }
        //send & rcv
        //send numTriples, sendmsg; receive numTriples rcvmsg
#ifndef NDEBUG
        std::cout << "GtzGate send p,g triples, size: " << sendmsg.size() << "\n";
#endif
        // std::thread t1([this, &sendmsg,&sendMacmsg]() {
        //     this->party->getNetwork().send(1 - this->myId(), sendmsg);
        //     this->party->getNetwork().send(1 - this->myId(), sendMacmsg);
        // });
        // std::thread t2([this, &rcvmsg,&rcvMacmsg]() {
        //     this->party->getNetwork().rcv(1 - this->myId(), &rcvmsg, rcvmsg.size());
        //     this->party->getNetwork().rcv(1 - this->myId(), &rcvMacmsg, rcvMacmsg.size());
        // });

        std::thread t1([this, &sendmsg, &sendMacmsg] {
            this->party().SendVecToOther(sendmsg);
            this->party().SendVecToOther(sendMacmsg);
        });
        std::thread t2([this, &rcvmsg, &rcvMacmsg, msgBytes, numTriples] {
            rcvmsg = this->party().template ReceiveVecFromOther<uint8_t>(msgBytes);
            rcvMacmsg = this->party().template ReceiveVecFromOther<ClearType>(numTriples * 2);
        });

        t1.join();
        t2.join();

        //#ifndef NDEBUG
        //            std::cout<<"sendmsg: ";printVector(sendmsg);
        //            std::cout<<"rcvmsg: "; printVector(rcvmsg);
        //#endif
        //compute
        bool alpha, beta, z, x_, y_;
        index_triple = 0;
        for (int i = 0; i < g.size(); ++i) {
            // parallel comparison
            for (int j = 0; j < u_len; ++j) {
                // compute u_p, u_g
                vec_loc = index_triple * 2 / 8;
                bit_loc = (index_triple * 2) % 8;
                index_triple++;
                alpha = std::bitset<sizeof(uint8_t) * 8>(sendmsg[vec_loc])[bit_loc] ^ std::bitset<sizeof(uint8_t) *
                    8>(rcvmsg[vec_loc])[bit_loc]; // alpha = p1 -a
                beta = std::bitset<sizeof(uint8_t) * 8>(sendmsg[vec_loc])[bit_loc + 1] ^ std::bitset<sizeof(uint8_t) *
                    8>(rcvmsg[vec_loc])[bit_loc + 1]; // beta = p2 - b
                x_ = p[i][2 * j], y_ = p[i][2 * j + 1]; //x_ -- p1, y_ -- p2
                z = c ^ (alpha & y_) ^ (beta & x_); // z = p1p2
                if (this->my_id() == 0) z ^= (alpha & beta);
                u_p[i][j] = z;
                vec_loc = index_triple * 2 / 8;
                bit_loc = (index_triple * 2) % 8;
                index_triple++;
                alpha = std::bitset<sizeof(uint8_t) * 8>(sendmsg[vec_loc])[bit_loc] ^ std::bitset<sizeof(uint8_t) *
                    8>(rcvmsg[vec_loc])[bit_loc]; // open p1 -a
                beta = std::bitset<sizeof(uint8_t) * 8>(sendmsg[vec_loc])[bit_loc + 1] ^ std::bitset<sizeof(uint8_t) *
                    8>(rcvmsg[vec_loc])[bit_loc + 1]; // open  p2 -b
                x_ = g[i][2 * j], y_ = p[i][2 * j + 1]; // x_ -- g1 y_ -- p2
                z = c ^ (alpha & y_) ^ (beta & x_); // z = p2g1
                if (this->my_id() == 0) z ^= (alpha & beta);
                u_g[i][j] = g[i][2 * j + 1] ^ z; // u_g = g2 + p2g1
            }
        }
        if (index_triple < numTriples) std::cout << "triples amount error\n";
        if (k % 2 == 1) {
            u_len += 1;
            for (int i = 0; i < p.size(); ++i) {
                u_p[i][u_len] = p[i][k - 1];
                u_g[i][u_len] = g[i][k - 1];
            }
        }
        auto ret = CarryOutAux(u_p, u_g, u_len); // u_len : bit length
        return ret;
    }
    else {
        //k<=1
        std::vector<bool> ret(g.size());
        for (int i = 0; i < ret.size(); ++i) {
            ret[i] = g[i][0];
        }
        return ret; // Actually only care g[..][0]
    }
}

} // namespace md_ml

#endif //GTZGATE_H
