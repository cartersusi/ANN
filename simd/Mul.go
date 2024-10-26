package simd

import (
	"math"
	"testing"
)

type (
	Operation func(left, right, result []float32) int
)

var (
	increment int32 = 1
	decrement int32 = math.MaxInt32
)

func Large(length int) []float32 {
	elements := make([]float32, length)
	for i := 0; i < length; i++ {
		elements[i] = float32(decrement)
		decrement--
	}
	return elements
}

func Small(length int) []float32 {
	elements := make([]float32, length)
	for i := 0; i < length; i++ {
		elements[i] = float32(increment)
		increment++
	}
	return elements
}

func Mul(left, right, result []float32) int {
	n := len(result)
	if m := len(left); m < n {
		n = m
	}
	if m := len(right); m < n {
		n = m
	}
	i := 0
	for ; n-i >= 4; i += 4 {
		result[i] = left[i] * right[i]
		result[i+1] = left[i+1] * right[i+1]
		result[i+2] = left[i+2] * right[i+2]
		result[i+3] = left[i+3] * right[i+3]
	}
	for ; i < n; i++ {
		result[i] = left[i] * right[i]
	}
	return n
}

func CheckSlice(t *testing.T, test, control []float32) bool {
	if len(test) != len(control) {
		t.Errorf("lengths not equal")
		return false
	}
	if cap(test) != cap(control) {
		t.Errorf("capacities not equal")
		return false
	}
	for i := 0; i < len(control); i++ {
		if test[i] != control[i] {
			t.Errorf("elements not equal")
			return false
		}
	}
	return true
}

func CheckOperation(t *testing.T, test, control Operation, left, right, result []float32) bool {
	testLeft := make([]float32, len(left), cap(left))
	copy(testLeft, left)
	testRight := make([]float32, len(right), cap(right))
	copy(testRight, right)
	testResult := make([]float32, len(result), cap(result))
	copy(testResult, result)
	if test(testLeft, testRight, testResult) != control(left, right, result) {
		t.Errorf("operation returned incorrect length")
		return false
	}
	if !CheckSlice(t, testLeft, left) {
		return false
	}
	if !CheckSlice(t, testRight, right) {
		return false
	}
	if !CheckSlice(t, testResult, result) {
		return false
	}
	return true
}
