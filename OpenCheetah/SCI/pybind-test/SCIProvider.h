#pragma once
#include <string>
#include <vector>
#include <library_fixed.h>

#if not defined(PARTY_ALICE) and not defined(PARTY_BOB)
    #define PARTY_ALICE
#endif

#if not defined(SCI_HE) and not defined(SCI_OT)
    #define SCI_HE
#endif

#if not defined(BIT_LENGTH)
    #define BIT_LENGTH 59
#endif

#ifdef PARTY_BOB
int party = 2;
#else
int party = 1;
#endif

int port = 31000;
std::string address = "127.0.0.1";
int num_threads = 4;
int32_t bitlength = BIT_LENGTH;

class CheetahProvider {

    int sf;

public:

    CheetahProvider(int sf): sf(sf) {}

    void startComputation() {
        StartComputation();
    }

    uint64_t endComputation() {
        return EndComputation(false);
    }

    int dbits() {
        return bitlength - sf;
    }

    std::pair<std::vector<uint64_t>, std::vector<uint8_t>> relu(std::vector<uint64_t> share, bool truncate) {
        long n = share.size();
        std::vector<uint64_t> r(n);
        std::vector<uint8_t> rd(n);
        ReluWithDRelu(n, share.data(), r.data(), rd.data(), sf, truncate);
        for (int i=0; i<n; i++) {
            rd[i] = party==1 ? rd[i] : (1-rd[i]);
        }
        return std::make_pair(std::move(r), std::move(rd));
    }

    std::vector<uint64_t> drelumul(std::vector<uint64_t> share, std::vector<uint8_t> drelu) {
        long n = share.size();
        for (long i=0; i<n; i++) drelu[i] = (party==1 ? drelu[i] : (1-drelu[i]));
        std::vector<uint64_t> ret(n);
        DReluMul(n, share.data(), drelu.data(), ret.data(), sf, false);
        return ret;
    }

    std::vector<uint64_t> truncate(std::vector<uint64_t> share) {
        long n = share.size();
        ScaleDown(n, share.data(), sf);
        return share;
    }

    std::vector<uint64_t> divide(std::vector<uint64_t> share, uint64_t divisor) {
        long n = share.size();
        Divide(n, share.data(), divisor);
        return share;
    }

    std::vector<uint64_t> sqrt(std::vector<uint64_t> share, int64_t scale_in, int64_t scale_out, bool inverse) {
        // This does not work.
        long n = share.size();
        std::vector<uint64_t> ret(n);
        Sqrt(1, n, scale_in, scale_out, bitlength, bitlength, inverse, share.data(), ret.data());
        return ret;
    }

    std::vector<uint64_t> elementwise_multiply(std::vector<uint64_t> share1, std::vector<uint64_t> share2) {
        long n = share1.size();
        std::vector<uint64_t> ret(n);
        ElemWiseSecretSharedVectorMult(n, share1.data(), share2.data(), ret.data());
        return ret;
    }

    std::vector<uint64_t> max(std::vector<uint64_t> share1, size_t result_count) {
        long n = share1.size(); assert(n % result_count == 0);
        std::vector<uint64_t> ret(result_count);
        MaxPool2D(result_count, n/result_count, bitlength, bitlength, (int64_t*)share1.data(), (int64_t*)ret.data());
        return ret;
    }

};