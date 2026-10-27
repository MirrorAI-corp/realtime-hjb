// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#include "hjb/io/npy_writer.hpp"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace hjb {

void writeNpy(const std::string& path,
              const double* data,
              const std::vector<int>& shape) {
    std::size_t total = 1;
    for (int d : shape) {
        if (d < 0) throw std::invalid_argument("writeNpy: negative dimension in shape");
        total *= static_cast<std::size_t>(d);
    }

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

    if (header.size() > 0xFFFF) throw std::invalid_argument("writeNpy: shape too long for an .npy v1.0 header");
    uint16_t header_len = static_cast<uint16_t>(header.size());

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("writeNpy: cannot open " + path);
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    f.write(magic, 6);
    char ver[2] = {1, 0};
    f.write(ver, 2);
    f.write(reinterpret_cast<const char*>(&header_len), 2);
    f.write(header.data(), header.size());

    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(sizeof(double) * total));
    if (!f) throw std::runtime_error("writeNpy: write failed for " + path);
}

} // namespace hjb
