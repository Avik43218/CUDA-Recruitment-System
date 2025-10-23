#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <cctype>
#include <cstdint>

using namespace std;
namespace fs = filesystem;

unordered_set<string> tokenize(const string &text) {
    unordered_set<string> terms;
    string word;
    for (char c : text) {
        if (isalpha(static_cast<unsigned char>(c)))
            word += tolower(static_cast<unsigned char>(c));
        else if (!word.empty()) {
            terms.insert(word);
            word.clear();
        }
    }
    if (!word.empty()) terms.insert(word);
    return terms;
}

vector<string> tokenize_sequence(const string &text) {
    vector<string> tokens;
    string word;
    for (char c : text) {
        if (isalpha(static_cast<unsigned char>(c)))
            word += tolower(static_cast<unsigned char>(c));
        else if (!word.empty()) {
            tokens.push_back(word);
            word.clear();
        }
    }
    if (!word.empty()) tokens.push_back(word);
    return tokens;
}


string read_file(const fs::path &path) {
    ifstream file(path);
    if (!file.is_open())
        throw runtime_error("Failed to open " + path.string());
    ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}


static inline uint32_t splitmix32(uint64_t x) {
    // Similar to SplitMix64 but truncated to 32 bits
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return static_cast<uint32_t>((x ^ (x >> 31)) & 0xFFFFFFFF);
}


int main() {
    cout << "Enter path to corpus folder: ";
    string folder_path;
    getline(cin, folder_path);

    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        cerr << "Error: invalid folder path.\n";
        return 1;
    }

    vector<string> file_paths;
    for (const auto &entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt")
            file_paths.push_back(entry.path().string());
    }

    if (file_paths.empty()) {
        cerr << "No .txt files found.\n";
        return 1;
    }

    // build vocabulary
    unordered_set<string> unique_terms;
    for (const auto &file : file_paths) {
        auto text = read_file(file);
        auto terms = tokenize(text);
        unique_terms.insert(terms.begin(), terms.end());
    }

    // assign deterministic 32-bit embedding per term 
    unordered_map<string, uint32_t> term_to_embedding;
    for (const auto &term : unique_terms) {
        uint64_t h = hash<string>{}(term);
        term_to_embedding[term] = splitmix32(h);
    }

    // Encode each document
    vector<vector<uint32_t>> encoded_docs;
    encoded_docs.reserve(file_paths.size());

    for (const auto &file : file_paths) {
        auto text = read_file(file);
        auto tokens = tokenize_sequence(text);

        vector<uint32_t> encoded;
        encoded.reserve(tokens.size());
        for (const auto &tok : tokens) {
            auto it = term_to_embedding.find(tok);
            if (it != term_to_embedding.end())
                encoded.push_back(it->second);
        }
        encoded_docs.push_back(std::move(encoded));
    }

    // Output results
    cout << "\n=== Vocabulary (" << term_to_embedding.size() << " terms) ===\n";
    for (const auto &p : term_to_embedding) {
        cout << left << setw(20) << p.first << " -> " << p.second << '\n';
    }

    return 0;
}
