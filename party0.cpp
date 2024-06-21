// By Boshi Yuan

#include "share/Spdz2kShare.h"
#include "protocols/Circuit.h"
#include "utils/print_vector.h"


using namespace std;
using namespace md_ml;

int main() {
    using ShrType = Spdz2kShare64;
    using ClearType = ShrType::ClearType;

    vector<ClearType> vec(65536, 1);

    PartyWithFakeOffline<ShrType> party(0, 2, 5050, "ResNet-18");
    Circuit<ShrType> circuit(party);

    auto a = circuit.input(0, 1, 65536);
    auto b = circuit.input(0, 65536, 1);
    auto c = circuit.multiply(a, b);
    auto d = circuit.output(c);
    circuit.addEndPoint(d);

    a->setInput(vec);
    b->setInput(vec);

    circuit.readOfflineFromFile();
    circuit.runOnlineWithBenckmark();

    circuit.printStats();

    return 0;
}
