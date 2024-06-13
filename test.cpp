// By Boshi Yuan

#include "fake-offline/FakeInputGate.h"
#include "fake-offline/FakeAddGate.h"
#include "fake-offline/FakeCircuit.h"
#include "share/Spdz2kShare.h"
#include "share/Mod2PowN.h"
#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeParty.h"

using namespace std;
using namespace md_ml;

int main() {
    using ShrType = Spdz2kShare64;
    using ClearType = ShrType::ClearType;

    FakeParty<ShrType, 2> party("ResNet-18");
    FakeCircuit<ShrType, 2> circuit(party);

    auto a = circuit.input(0, 1, 1);
    auto b = circuit.input(1, 1, 1);
    // auto c = circuit.add(a, b);
    // auto d = circuit.subtract(c, a);
    auto e = circuit.multiply(a, b);

    circuit.addEndpoint(e);
    circuit.runOffline();

    return 0;
}
