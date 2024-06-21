// By Boshi Yuan

#ifndef MULTIPLYGATE_H
#define MULTIPLYGATE_H

#include <memory>
#include <vector>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"

namespace md_ml {

template <IsSpdz2kShare ShrType>
class MultiplyGate : public Gate<ShrType> {

};

} // md_ml

#endif //MULTIPLYGATE_H
