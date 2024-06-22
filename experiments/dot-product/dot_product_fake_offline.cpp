// By Boshi Yuan

#include "dot_product_config.h"

#include "fake-offline/FakeInputGate.h"
#include "fake-offline/FakeAddGate.h"
#include "fake-offline/FakeCircuit.h"
#include "share/Spdz2kShare.h"
#include "share/Mod2PowN.h"
#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeParty.h"

using namespace std;
using namespace md_ml;
using namespace md_ml::experiments::dot_product;


int main() {
    using ShrType = Spdz2kShare64;

    FakeParty<ShrType, 2> party(kJobName);
    FakeCircuit<ShrType, 2> circuit(party);

    auto a = circuit.input(0, 1, dim);
    auto b = circuit.input(0, dim, 1);
    auto c = circuit.multiply(a, b);
    auto d = circuit.output(c);

    circuit.addEndpoint(d);
    circuit.runOffline();

    return 0;
}
