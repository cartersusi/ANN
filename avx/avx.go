//go:build amd64
// +build amd64

package avx

func Supported() bool

//func AddFloat32(left, right, result []float32) int

//func DivFloat32(left, right, result []float32) int

func DotProductFloat32(left, right []float32, result float32) float32

//func SubFloat32(left, right, result []float32) int
