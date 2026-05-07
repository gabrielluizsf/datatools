package datatools

import (
	"fmt"
	"testing"

	"github.com/atendi9/capivara/assert"
)

func TestSoftmaxMatrix(t *testing.T) {
	input := [][]float64{
		{1.0, 2.0, 3.0},
		{0.0, 0.0},
	}

	expected := [][]float64{
		{0.09003057317038046, 0.24472847105479764, 0.6652409557748218},
		{0.5, 0.5},
	}

	result := SoftmaxMatrix(input)

	assert.Equal(t, fmt.Sprint(result), fmt.Sprint(expected))

	for _, row := range result {
		sum := 0.0
		for _, val := range row {
			sum += val
		}

		if sum < 0.999999 || sum > 1.000001 {
			t.Errorf("Expected row sum to be 1.0, got %f", sum)
		}
	}
}
