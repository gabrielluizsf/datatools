package datatools

import "sync"

// LinearTransformation performs a linear transformation on an input vector.
// It computes the result using the formula:
//
// y = xWᵀ + b
//
// Where:
// 	- x is the input vector (1xn)
// 	- W is the weight matrix (mxn)
// 	- b is the bias vector (1xm)
// 	- y is the output vector (1xm)
//
// Each element y[i] is calculated as:
//
// y[i] = Σ (input[j] * weights[i][j]) + bias[i]
func LinearTransformation(
	input []float64,
	weights [][]float64,
	bias []float64,
) []float64 {
	var wg sync.WaitGroup
	output := make([]float64, len(weights))
	for i := range weights {
		wg.Go(func() {
			sum := 0.0
			for j := range input {
				sum += input[j] * weights[i][j]
			}
			output[i] = sum + bias[i]
		})
	}
	wg.Wait()
	return output
}
