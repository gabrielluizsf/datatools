package datatools

import (
	"math"
	"sync"
)

// SoftmaxMatrix applies the Softmax function to each row of a 2D matrix.
// 	It uses a stable implementation by subtracting the maximum value of each row
// 	to prevent numerical overflow during exponentiation.
// 	The computation is performed concurrently for each row.
func SoftmaxMatrix(matrix [][]float64) [][]float64 {
	result := make([][]float64, len(matrix))
	var wg sync.WaitGroup

	for i := range matrix {
		wg.Go(func() {
			result[i] = make([]float64, len(matrix[i]))

			max := matrix[i][0]
			for _, v := range matrix[i] {
				if v > max {
					max = v
				}
			}

			sum := 0.0
			for j, v := range matrix[i] {
				result[i][j] = math.Exp(v - max)
				sum += result[i][j]
			}

			for j := range result[i] {
				result[i][j] /= sum
			}
		})
	}

	wg.Wait()
	return result
}
