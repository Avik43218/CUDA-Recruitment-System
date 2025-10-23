// preprocess.hpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

struct CorpusData {
    std::vector<uint32_t> flat_docs;
    std::vector<int> doc_offsets;
    int vocab_size;
    int num_docs;
    std::unordered_map<std::string, uint32_t> term_to_id;
    std::vector<std::string> id_to_term;
};

CorpusData preprocess_corpus(const std::string &folder_path);
