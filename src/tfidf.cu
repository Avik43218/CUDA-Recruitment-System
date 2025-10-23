// tfidf_gpu.cu
#include "preprocess.hpp"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

__global__ void compute_tf_df(
    const uint32_t *docs, const int *doc_offsets, int num_docs,
    uint32_t *tf_counts, uint32_t *df_counts, int vocab_size)
{
    int doc_id = blockIdx.x;
    if (doc_id >= num_docs) return;

    int start = doc_offsets[doc_id];
    int end   = doc_offsets[doc_id + 1];

    extern __shared__ uint8_t term_flags[];
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x)
        term_flags[i] = 0;
    __syncthreads();

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        uint32_t term = docs[i];
        if (term < vocab_size) {
            atomicAdd(&tf_counts[doc_id * vocab_size + term], 1);
            term_flags[term] = 1;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int t = 0; t < vocab_size; ++t)
            if (term_flags[t]) atomicAdd(&df_counts[t], 1);
    }
}

__global__ void compute_tfidf(
    const uint32_t *tf_counts, const uint32_t *df_counts,
    float *tfidf, int num_docs, int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_docs * vocab_size;
    if (idx >= total) return;

    int term_id = idx % vocab_size;
    int df = df_counts[term_id];
    if (df == 0) return;

    float tf = static_cast<float>(tf_counts[idx]);
    float idf = logf(static_cast<float>(num_docs) / df);
    tfidf[idx] = tf * idf;
}

// -------- GPU runner --------

void run_tfidf_on_gpu(const CorpusData &data) {
    int num_docs = data.num_docs;
    int vocab_size = data.vocab_size;
    const auto &flat_docs = data.flat_docs;
    const auto &doc_offsets = data.doc_offsets;

    cout << "Launching GPU kernels... (docs=" << num_docs
              << ", vocab=" << vocab_size << ")\n";

    uint32_t *d_docs, *d_tf, *d_df;
    int *d_offsets;
    float *d_tfidf;

    cudaMalloc(&d_docs, flat_docs.size() * sizeof(uint32_t));
    cudaMalloc(&d_offsets, doc_offsets.size() * sizeof(int));
    cudaMalloc(&d_tf, num_docs * vocab_size * sizeof(uint32_t));
    cudaMalloc(&d_df, vocab_size * sizeof(uint32_t));
    cudaMalloc(&d_tfidf, num_docs * vocab_size * sizeof(float));

    cudaMemcpy(d_docs, flat_docs.data(), flat_docs.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, doc_offsets.data(), doc_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_tf, 0, num_docs * vocab_size * sizeof(uint32_t));
    cudaMemset(d_df, 0, vocab_size * sizeof(uint32_t));

    compute_tf_df<<<num_docs, 128, vocab_size * sizeof(uint8_t)>>>(
        d_docs, d_offsets, num_docs, d_tf, d_df, vocab_size);
    cudaDeviceSynchronize();

    int threads = 128;
    int blocks = (num_docs * vocab_size + threads - 1) / threads;
    compute_tfidf<<<blocks, threads>>>(d_tf, d_df, d_tfidf, num_docs, vocab_size);
    cudaDeviceSynchronize();

    vector<float> h_tfidf(num_docs * vocab_size);
    cudaMemcpy(h_tfidf.data(), d_tfidf, h_tfidf.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "\n=== TF-IDF Matrix (first 10 terms per doc) ===\n";
    for (int d = 0; d < num_docs; ++d) {
        cout << "Doc " << d << ": [ ";
        for (int t = 0; t < min(10, vocab_size); ++t)
            cout << h_tfidf[d * vocab_size + t] << " ";
        cout << "... ]\n";
    }

    cudaFree(d_docs);
    cudaFree(d_offsets);
    cudaFree(d_tf);
    cudaFree(d_df);
    cudaFree(d_tfidf);
}

// -------- Main entry --------
int main() {
    cout << "Enter corpus folder path: ";
    string folder;
    getline(cin, folder);

    try {
        CorpusData data = preprocess_corpus(folder);
        run_tfidf_on_gpu(data);
    } catch (const std::exception &e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    cout << "\nTF-IDF complete.\n";
    return 0;
}
