package datatools

import (
	"testing"

	"github.com/atendi9/capivara/assert"
)

func TestBellmanEquation(t *testing.T) {
	reward := 10.0
	gamma := 0.9
	nextMaxQ := 20.0
	expected := 28.0

	result := BellmanEquation(reward, gamma, nextMaxQ)

	assert.Equal(t, result, expected)
}
