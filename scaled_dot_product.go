package datatools

import (
	"math"
	"sync"
)

// ScaledDotProductAttention computes the attention mechanism used in Transformer models.
// It maps a query and a set of key-value pairs to an output.
//
//		The attention score is calculated in three steps:
//
//	 	1. Compute the similarity between queries and keys:
//
//	    		scores = QK^T
//
//	 	2. Scale the scores by the square root of the key dimension:
//
//	    		scaled_scores = scores / √d_k
//
//	 	3. Apply softmax to obtain attention weights and multiply by V:
//
//	    		Attention(Q, K, V) = softmax(scaled_scores) · V
//
// Full formula:
//
//	:contentReference[oaicite:0]{index=0}
//
// Where:
//   - Q: Query matrix
//   - K: Key matrix
//   - V: Value matrix
//   - d_k: Dimension of the key vectors
//
// Parameters:
//   - query: A matrix representing the queries of shape [n_queries, d_k].
//   - key: A matrix representing the keys of shape [n_keys, d_k].
//   - value: A matrix representing the values of shape [n_keys, d_v].
//
// Returns:
//   - A matrix representing the attention output of shape [n_queries, d_v].
func ScaledDotProductAttention(
	query, key, value [][]float64,
) [][]float64 {
	dim := float64(len(key[0]))
	scale := math.Sqrt(dim)
	var wg sync.WaitGroup

	scores := make([][]float64, len(query))
	for i := range query {
		wg.Go(func() {
			scores[i] = make([]float64, len(key))
			for j := range key {
				dot := 0.0
				for k := range query[i] {
					dot += query[i][k] * key[j][k]
				}
				scores[i][j] = dot / scale
			}
		})
	}
	wg.Wait()

	attentionWeights := SoftmaxMatrix(scores)
	result := make([][]float64, len(attentionWeights))
	for i := range attentionWeights {
		wg.Go(func() {
			result[i] = make([]float64, len(value[0]))
			for j := range value[0] {
				sum := 0.0
				for k := range attentionWeights[i] {
					sum += attentionWeights[i][k] * value[k][j]
				}
				result[i][j] = sum
			}
		})
	}
	wg.Wait()

	return result
}
