// Python bindings for FastFilter's four-wise binary fuse algorithm.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "fastfilter/4wise_xor_binary_fuse_filter_lowmem.h"
namespace py = pybind11;
struct SeededHash {
    static thread_local uint64_t next_seed;
    uint64_t seed;
    SeededHash(): seed(next_seed) { next_seed += UINT64_C(0x9e3779b97f4a7c15); }
    uint64_t operator()(uint64_t key) const { return hashing::SimpleMixSplit::murmur64(key+seed); }
};
thread_local uint64_t SeededHash::next_seed = 0;
using Filter = xorbinaryfusefilter_lowmem4wise::XorBinaryFuseFilter<uint64_t,uint8_t,SeededHash>;
using Keys = py::array_t<uint64_t,py::array::c_style|py::array::forcecast>;
py::tuple build(Keys input, uint64_t seed) {
    if(input.ndim()!=1 || input.size()==0 || input.size()>100000000) throw std::invalid_argument("Invalid key count");
    vector<uint64_t> keys(input.data(),input.data()+input.size());
    std::sort(keys.begin(),keys.end());
    if(std::adjacent_find(keys.begin(),keys.end())!=keys.end()) throw std::invalid_argument("Duplicate keys");
    SeededHash::next_seed = seed;
    Filter filter(keys.size());
    filter.AddAll(keys,0,keys.size());
    for(auto key:keys) if(filter.Contain(key)!=0) throw std::runtime_error("Filter false negative");
    py::dict meta;
    meta["arity"]=4; meta["seed"]=filter.hasher->seed;
    meta["segment_length"]=filter.segmentLength; meta["segment_count"]=filter.segmentCount;
    meta["array_length"]=filter.arrayLength;
    return py::make_tuple(meta,py::bytes((char*)filter.fingerprints,filter.arrayLength));
}
py::array_t<bool> query(Keys keys, py::dict meta, py::bytes raw) {
    auto arity=meta["arity"].cast<size_t>();
    auto seed=meta["seed"].cast<uint64_t>();
    auto length=meta["segment_length"].cast<size_t>();
    auto count=meta["segment_count"].cast<size_t>();
    auto total=meta["array_length"].cast<size_t>();
    std::string data=raw;
    if(arity!=4 || length==0 || length>(1<<18) || (length&(length-1)) || count==0 || count>100000000 || total!=data.size() || total!=(count+3)*length)
        throw std::invalid_argument("Invalid filter geometry");
    if(keys.ndim()!=1) throw std::invalid_argument("Keys must be flat");
    py::array_t<bool> output(keys.size());
    auto out=output.mutable_unchecked<1>();
    for(ssize_t j=0;j<keys.size();j++) {
        uint64_t hash=hashing::SimpleMixSplit::murmur64(keys.data()[j]+seed);
        size_t base=(__uint128_t(hash)*__uint128_t(count*length))>>64;
        uint8_t f=uint8_t(hash);
        for(int h=0;h<4;h++) {
            size_t pos=base+h*length;
            if(h>0) pos^=(hash>>((h-1)*16))&(length-1);
            f^=uint8_t(data[pos]);
        }
        out(j)=f==0;
    }
    return output;
}
PYBIND11_MODULE(_bfuse,m) { m.def("build",&build);m.def("query",&query); }
