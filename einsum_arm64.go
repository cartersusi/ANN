//go:build arm64

package ann

import (
	"fmt"
	"math"
	"time"
)

func getSimdImplementation() SimdInterface {
	return &FallbackImplementation{}
}

type FallbackImplementation struct{}

func (f *FallbackImplementation) Einsum(query []float32, database [][]float32, nxdim int) []float32 {
	return np_einsum_large(query, database)
}

func CheckEinsumARM(query []float32, database [][]float32, nxdim int) {
	start_time := time.Now()
	large := np_einsum_large(query, database)
	fmt.Println("Time taken for Large Einsum:", time.Since(start_time))

	start_time = time.Now()
	medium := np_einsum(query, database)
	fmt.Println("Time taken for Medium Einsum:", time.Since(start_time))

	start_time = time.Now()
	small := einsum_small(query, database)
	fmt.Println("Time taken for Small Einsum:", time.Since(start_time))

	tolarance := float32(0.1)

	for i := 0; i < len(small); i++ {
		if math.Abs(float64(small[i]-medium[i])) > float64(tolarance) {
			panic(fmt.Sprintf("Small Einsum and Medium Einsum are not equal %f %f", small[i], medium[i]))
		}
	}

	for i := 0; i < len(large); i++ {
		if math.Abs(float64(large[i]-medium[i])) > float64(tolarance) {
			panic(fmt.Sprintf("Large Einsum and Medium Einsum are not equal %f %f", large[i], medium[i]))
		}
	}
}
