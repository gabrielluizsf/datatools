package datatools

import (
	"testing"

	"github.com/atendi9/capivara/assert"
)

func TestScaledDotProductAttention(t *testing.T) {
	t.Run("Identity Attention", func(t *testing.T) {
		query := [][]float64{{1.0, 0.0}, {0.0, 1.0}}
		key := [][]float64{{1.0, 0.0}, {0.0, 1.0}}
		value := [][]float64{{10.0, 20.0}, {30.0, 40.0}}

		result := ScaledDotProductAttention(query, key, value)

		assert.Equal(t, len(result), 2)
		assert.Equal(t, len(result[0]), 2)

		expectedFirstVal := 16.60476901346686
		assert.Equal(t, result[0][0], expectedFirstVal)
	})

	t.Run("Dimension Mismatch Handling", func(t *testing.T) {
		query := [][]float64{{1.0, 0.5}}
		key := [][]float64{{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}
		value := [][]float64{{1.0}, {2.0}, {3.0}}

		result := ScaledDotProductAttention(query, key, value)

		assert.LengthSlice(t, 1, result)
		assert.LengthSlice(t, 1, result[0])
	})
}
