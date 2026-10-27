// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#include "hjb/io/npy_writer.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace hjb {

void writeNpy(const std::string& path,
              const double* data,
              const std::vector<int>& shape) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "writeNpy: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    f.write(magic, 6);
    char ver[2] = {1, 0};
    f.write(ver, 2);

    std::ostringstream shape_str;
    shape_str << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str << shape[i];
        if (shape.size() == 1 || i + 1 < shape.size()) shape_str << ", ";
    }
    shape_str << ")";

    std::ostringstream hdr;
    hdr << "{'descr': '<f8', 'fortran_order': False, 'shape': "
        << shape_str.str() << ", }";
    std::string header = hdr.str();

    int base = 6 + 2 + 2 + static_cast<int>(header.size()) + 1; // +1 for '\n'
    int pad = (64 - (base % 64)) % 64;
    header.append(pad, ' ');
    header.push_back('\n');

    uint16_t header_len = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&header_len), 2);
    f.write(header.data(), header.size());

    std::size_t total = 1;
    for (int d : shape) total *= static_cast<std::size_t>(d);
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(sizeof(double) * total));
}

} // namespace hjb
