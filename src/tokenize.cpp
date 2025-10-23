// preprocess.cpp
#include "preprocess.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <filesystem>
#include <cctype>

using namespace std;
namespace fs = filesystem;

static inline uint32_t splitmix32(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return static_cast<uint32_t>((x ^ (x >> 31)) & 0xFFFFFFFF);
}

static unordered_set<string> tokenize(const string &text) {
     unordered_set<string> terms;
     string word;
    for (char c : text) {
        if ( isalpha(static_cast<unsigned char>(c)))
            word +=  tolower(static_cast<unsigned char>(c));
        else if (!word.empty()) {
            terms.insert(word);
            word.clear();
        }
    }
    if (!word.empty()) terms.insert(word);
    return terms;
}

static  vector<string> tokenize_sequence(const string &text) {
     vector<string> tokens;
     string word;
    for (char c : text) {
        if ( isalpha(static_cast<unsigned char>(c)))
            word += tolower(static_cast<unsigned char>(c));
        else if (!word.empty()) {
            tokens.push_back(word);
            word.clear();
        }
    }
    if (!word.empty()) tokens.push_back(word);
    return tokens;
}

static string read_file(const fs::path &path) {
     ifstream file(path);
    if (!file.is_open())
        throw runtime_error("Failed to open " + path.string());
     ostringstream buf;
    buf << file.rdbuf();
    return buf.str();
}

CorpusData preprocess_corpus(const string &folder_path) {
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path))
        throw runtime_error("Invalid folder path: " + folder_path);

     vector<string> file_paths;
    for (const auto &entry : fs::directory_iterator(folder_path))
        if (entry.is_regular_file() && entry.path().extension() == ".txt")
            file_paths.push_back(entry.path().string());

    if (file_paths.empty())
        throw runtime_error("No .txt files found in folder.");

     unordered_set<string> unique_terms;
    for (const auto &file : file_paths) {
        auto text = read_file(file);
        auto terms = tokenize(text);
        unique_terms.insert(terms.begin(), terms.end());
    }

     unordered_map<string, uint32_t> term_to_id;
     vector<string> id_to_term;
    id_to_term.reserve(unique_terms.size());
    int idx = 0;
    for (const auto &term : unique_terms) {
        term_to_id[term] = idx++;
        id_to_term.push_back(term);
    }

     vector<uint32_t> flat_docs;
     vector<int> doc_offsets = {0};
    for (const auto &file : file_paths) {
        auto text = read_file(file);
        auto tokens = tokenize_sequence(text);
        for (const auto &tok : tokens) {
            auto it = term_to_id.find(tok);
            if (it != term_to_id.end())
                flat_docs.push_back(it->second);
        }
        doc_offsets.push_back(flat_docs.size());
    }

    CorpusData data;
    data.flat_docs =  move(flat_docs);
    data.doc_offsets = move(doc_offsets);
    data.vocab_size = static_cast<int>(term_to_id.size());
    data.num_docs = static_cast<int>(file_paths.size());
    data.term_to_id = move(term_to_id);
    data.id_to_term = move(id_to_term);
    return data;
}
