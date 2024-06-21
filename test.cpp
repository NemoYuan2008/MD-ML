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

    FakeParty<ShrType, 2> party("ResNet-18");
    FakeCircuit<ShrType, 2> circuit(party);

    auto a = circuit.input(0, 1, 65536);
    auto b = circuit.input(0, 65536, 1);
    auto c = circuit.multiply(a, b);
    auto d = circuit.output(c);

    circuit.addEndpoint(d);
    circuit.runOffline();

    return 0;
}
