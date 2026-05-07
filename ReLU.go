package datatools

import "sync"

// ReLU applies the Rectified Linear Unit activation function to a slice of float64.
//
//	It uses the formula and processes elements concurrently:
//		f(x) = max(0, x)
//
// For more information on ReLU, see: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
func ReLU(input []float64) []float64 {
	var wg sync.WaitGroup
	output := make([]float64, len(input))

	for i, val := range input {
		wg.Go(func() {
			if val > 0 {
				output[i] = val
			} else {
				output[i] = 0
			}
		})
	}

	wg.Wait()
	return output
}
