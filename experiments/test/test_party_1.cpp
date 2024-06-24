// By Boshi Yuan

#include "share/Spdz2kShare.h"
#include "protocols/Circuit.h"
#include "utils/print_vector.h"


using namespace std;
using namespace md_ml;

int main() {
    using ShrType = Spdz2kShare64;
    using ClearType = ShrType::ClearType;

    PartyWithFakeOffline<ShrType> party(1, 2, 5050, "test");
    Circuit<ShrType> circuit(party);

    // // Tests for truncation correctness
    // auto a = circuit.input(0, 1, 1);
    // auto b = circuit.input(0, 1, 1);
    // auto c = circuit.multiplyTrunc(a, b);
    // auto d = circuit.output(c);
    // circuit.addEndPoint(d);

    // circuit.readOfflineFromFile();
    // circuit.runOnlineWithBenckmark();

    // circuit.printStats();

    return 0;
}
