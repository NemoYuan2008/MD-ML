// By Boshi Yuan

#include "share/Spdz2kShare.h"
#include "protocols/Circuit.h"
#include "utils/print_vector.h"


using namespace std;
using namespace md_ml;

int main() {
    using ShrType = Spdz2kShare64;
    using ClearType = ShrType::ClearType;

    PartyWithFakeOffline<ShrType> party(1, 2, 5050, "ResNet-18");
    Circuit<ShrType> circuit(party);

    // Use the member function from circuit
    auto a = circuit.input(0, 1, 5);
    auto b = circuit.input(0, 1, 5);
    auto c = circuit.add(a, b);
    auto d = circuit.output(c);

    d->readOfflineFromFile();
    d->RunOnline();

    PrintVector(d->getClear());

    return 0;
}
