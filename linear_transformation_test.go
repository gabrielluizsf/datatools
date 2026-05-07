package datatools

import (
	"fmt"
	"testing"

	"github.com/atendi9/capivara/assert"
)

func TestLinearTransformation(t *testing.T) {
	input := []float64{1.0, 2.0}
	weights := [][]float64{
		{0.5, 0.1},
		{0.2, 0.8},
	}
	bias := []float64{0.5, 1.0}

	expected := []float64{1.2, 2.8}

	result := LinearTransformation(input, weights, bias)

	assert.Equal(t, fmt.Sprint(result), fmt.Sprint(expected))
}
