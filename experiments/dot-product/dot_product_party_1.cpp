// By Boshi Yuan

#include "dot_product_config.h"

#include "share/Spdz2kShare.h"
#include "protocols/Circuit.h"
#include "utils/print_vector.h"

using namespace std;
using namespace md_ml;
using namespace md_ml::experiments::dot_product;

int main() {
    using ShrType = Spdz2kShare64;
    using ClearType = ShrType::ClearType;

    PartyWithFakeOffline<ShrType> party(1, 2, 5050, kJobName);
    Circuit<ShrType> circuit(party);

    auto a = circuit.input(0, 1, dim);
    auto b = circuit.input(0, dim, 1);
    auto c = circuit.multiply(a, b);
    auto d = circuit.output(c);
    circuit.addEndpoint(d);

    circuit.readOfflineFromFile();
    circuit.runOnlineWithBenckmark();

    circuit.printStats();

    return 0;
}
