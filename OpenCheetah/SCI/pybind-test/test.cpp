#include "SCIProvider.h"



int main() {
    SCIProvider provider;
    provider.startComputation();
    int n = 5;
    std::vector<double> r;
    for (int i=0; i<n; i++) {
        if (i%2==1) r.push_back((double)(i)*5);
        else {
            if (party == 1) r.push_back(0);
            else r.push_back(-i*10);
        }
        std::cout << "input r[" << i << "] = " << r[i] << std::endl;
    }
    auto result = provider.relu(r);
    for (int i=0; i<n; i++) {
        std::cout << "rel[" << i << "]=" << result.first[i] << ", d=" << (int)(result.second[i]) << std::endl;
    }
    provider.endComputation();
    return 0;
}