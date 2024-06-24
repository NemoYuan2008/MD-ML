// By Boshi Yuan

#include "share/Spdz2kShare.h"
#include "protocols/Circuit.h"
#include "utils/print_vector.h"
#include "utils/fixed_point.h"

using namespace std;
using namespace md_ml;

int main() {
    using ShrType = Spdz2kShare64;
    using ClearType = ShrType::ClearType;

    vector<ClearType> vec{};
    vec.push_back(double2fix<ClearType>(1.5));

    PartyWithFakeOffline<ShrType> party(0, 2, 5050, "test");
    Circuit<ShrType> circuit(party);

    // // Tests for truncation correctness
    // auto a = circuit.input(0, 1, 1);
    // auto b = circuit.input(0, 1, 1);
    // auto c = circuit.multiplyTrunc(a, b);
    // auto d = circuit.output(c);
    // circuit.addEndPoint(d);
    //
    // a->setInput(vec);
    // b->setInput(vec);
    //
    // circuit.readOfflineFromFile();
    // circuit.runOnlineWithBenckmark();
    //
    // auto o = d->getClear();
    // cout << "output: " << fix2double<ClearType>(o[0]) << endl;



    std::vector<ClearType> x_in{
        1, 2, 3, 1, 1, 1, 2, 2, 2,
        2, 1, 3, 4, 2, 1, 2, 2, 1,
    };
    std::vector<ClearType> y_in{
        0, 1, 2, 3, 2, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 1, 1, 1,
    };

    circuit.printStats();

    return 0;
}
