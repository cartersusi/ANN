//go:build amd64

package ann

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cartersusi/ann/avx"
)

func getSimdImplementation() SimdInterface {
	return &AvxImplementation{}
}

type AvxImplementation struct{}

func (a *AvxImplementation) Einsum(query []float32, database [][]float32, nxdim int) []float32 {
	if avx.Supported() {
		return np_einsum_large_simd(query, database)
	}
	return np_einsum_large(query, database)
}

func np_einsum_large_simd(query []float32, database [][]float32) []float32 {
	dbLen := len(database)
	queryLen := len(query)
	if queryLen == 0 || dbLen == 0 {
		return make([]float32, dbLen)
	}

	scores := alignedFloat32Slice(dbLen)
	numCPU := runtime.GOMAXPROCS(0)
	chunkSize := getOptimalChunkSize(dbLen, queryLen)

	resultBufferPool := sync.Pool{
		New: func() interface{} {
			return make([]float32, queryLen)
		},
	}

	var wg sync.WaitGroup
	workers := make(chan struct{}, numCPU)
	var activeWorkers int32

	for chunk := 0; chunk < dbLen; chunk += chunkSize {
		workers <- struct{}{}
		wg.Add(1)
		atomic.AddInt32(&activeWorkers, 1)

		go func(startIdx, endIdx int) {
			defer func() {
				<-workers
				atomic.AddInt32(&activeWorkers, -1)
				wg.Done()
			}()

			result := resultBufferPool.Get().([]float32)
			defer resultBufferPool.Put(result)

			endIdx = int(math.Min(float64(endIdx), float64(dbLen)))
			if endIdx < dbLen {
				_ = database[endIdx]
			}

			for i := startIdx; i < endIdx; i += outerBlockSize {
				iEnd := int(math.Min(float64(i+outerBlockSize), float64(endIdx)))

				for ii := i; ii < iEnd; ii++ {
					d_vec := database[ii]

					result := float32(0)
					scores[ii] = avx.DotProduct(query, d_vec, result)
				}
			}
		}(chunk, chunk+chunkSize)
	}

	wg.Wait()
	return scores
}

func CheckEinsumAMD(query []float32, database [][]float32, nxdim int) {
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

	if avx.Supported() {
		start_time = time.Now()
		simd_large := np_einsum_large_simd(query, database)
		fmt.Println("Time taken for Large Einsum SIMD:", time.Since(start_time))

		for i := 0; i < len(large); i++ {
			if math.Abs(float64(large[i]-simd_large[i])) > float64(tolarance) {
				panic(fmt.Sprintf("Large Einsum SIMD and Large Einsum are not equal %f %f", large[i], simd_large[i]))
			}
		}
	} else {
		fmt.Println("AVX not supported")
		for i := 0; i < len(large); i++ {
			if math.Abs(float64(large[i]-medium[i])) > float64(tolarance) {
				panic(fmt.Sprintf("Large Einsum and Medium Einsum are not equal %f %f", large[i], medium[i]))
			}
		}
	}
}
