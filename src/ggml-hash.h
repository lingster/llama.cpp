#ifndef GGML_HASH_H
#define GGML_HASH_H

#include <string>
#include <cstdint>

// Simple FNV-1a hash implementation
static inline std::string ggml_hash_string(const char* data, size_t size) {
    const uint64_t FNV_PRIME = 1099511628211ULL;
    const uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;

    uint64_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= FNV_PRIME;
    }

    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016llx", (unsigned long long)hash);
    return std::string(hash_str);
}

#endif // GGML_HASH_H