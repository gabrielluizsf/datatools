package datatools

import (
	"testing"

	"github.com/atendi9/capivara/assert"
)

func TestReLU(t *testing.T) {
	input := []float64{-1.5, 0, 2.0, -0.1, 5.5}
	expected := []float64{0, 0, 2.0, 0, 5.5}

	result := ReLU(input)
	assert.LengthSlice(t, len(expected), result)

	for i := range result {
		assert.Equal(t, expected[i], result[i])
	}
}

func TestReLU_AllNegative(t *testing.T) {
	input := []float64{-10, -20, -30}
	expected := []float64{0, 0, 0}

	result := ReLU(input)

	for i := range result {
		assert.Equal(t, expected[i], result[i])
	}
}

func TestReLU_Empty(t *testing.T) {
	input := []float64{}
	result := ReLU(input)

	assert.Equal(t, len(result), 0)
}
