package ann

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

const (
	outerBlockSize = 32
	innerBlockSize = 16
	cacheLineSize  = 64
)

type SimdInterface interface {
	Einsum(query []float32, database [][]float32, nxdim int) []float32
}

func Einsum(query []float32, database [][]float32, nxdim int) []float32 {
	if len(query) == 0 || len(database) == 0 {
		return nil
	}

	if nxdim < small_threshold {
		return einsum_small(query, database)
	} else if nxdim < medium_threshold {
		return np_einsum(query, database)
	}

	impl := getSimdImplementation()
	return impl.Einsum(query, database, nxdim)
}

func alignedFloat32Slice(size int) []float32 {
	rawSlice := make([]float32, size+cacheLineSize/4)
	addr := uintptr(unsafe.Pointer(&rawSlice[0]))
	offset := (cacheLineSize - (addr % cacheLineSize)) / 4
	return rawSlice[int(offset) : int(offset)+size]
}

type ScoresAccumulator struct {
	scores []float32
	pad    [cacheLineSize]byte
}

func np_einsum_large(query []float32, database [][]float32) []float32 {
	dbLen := len(database)
	queryLen := len(query)

	if queryLen == 0 || dbLen == 0 {
		return make([]float32, dbLen)
	}

	scores := alignedFloat32Slice(dbLen)

	numCPU := runtime.GOMAXPROCS(0)
	chunkSize := getOptimalChunkSize(dbLen, queryLen)

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

			endIdx = int(math.Min(float64(endIdx), float64(dbLen)))

			if endIdx < dbLen {
				_ = database[endIdx]
			}

			for i := startIdx; i < endIdx; i += outerBlockSize {
				iEnd := int(math.Min(float64(i+outerBlockSize), float64(endIdx)))

				for k := 0; k < queryLen; k += innerBlockSize {
					kEnd := int(math.Min(float64(k+innerBlockSize), float64(queryLen)))

					for ii := i; ii < iEnd; ii++ {
						d_vec := database[ii]
						dot := scores[ii]

						for kk := k; kk < kEnd-15; kk += 16 {
							dot += query[kk] * d_vec[kk]
							dot += query[kk+1] * d_vec[kk+1]
							dot += query[kk+2] * d_vec[kk+2]
							dot += query[kk+3] * d_vec[kk+3]
							dot += query[kk+4] * d_vec[kk+4]
							dot += query[kk+5] * d_vec[kk+5]
							dot += query[kk+6] * d_vec[kk+6]
							dot += query[kk+7] * d_vec[kk+7]
							dot += query[kk+8] * d_vec[kk+8]
							dot += query[kk+9] * d_vec[kk+9]
							dot += query[kk+10] * d_vec[kk+10]
							dot += query[kk+11] * d_vec[kk+11]
							dot += query[kk+12] * d_vec[kk+12]
							dot += query[kk+13] * d_vec[kk+13]
							dot += query[kk+14] * d_vec[kk+14]
							dot += query[kk+15] * d_vec[kk+15]
						}

						for kk := kEnd - (kEnd % 16); kk < kEnd; kk++ {
							dot += query[kk] * d_vec[kk]
						}

						scores[ii] = dot
					}
				}
			}
		}(chunk, chunk+chunkSize)
	}

	wg.Wait()
	return scores
}

func getOptimalChunkSize(dbSize, vectorSize int) int {
	numCPU := runtime.GOMAXPROCS(0)
	chunkSize := (dbSize + numCPU - 1) / numCPU

	if vectorSize > 1024 {
		chunkSize = int(float64(chunkSize) * 0.75)
	}

	minChunk := 4
	if chunkSize < minChunk {
		chunkSize = minChunk
	}

	return ((chunkSize + outerBlockSize - 1) / outerBlockSize) * outerBlockSize
}

func np_einsum(query []float32, database [][]float32) []float32 {
	const (
		outerBlockSize = 16
		innerBlockSize = 8
	)

	dbLen := len(database)
	queryLen := len(query)
	scores := make([]float32, dbLen)

	if queryLen == 0 || dbLen == 0 {
		return scores
	}

	for i := 0; i < dbLen; i += outerBlockSize {
		iEnd := int(math.Min(float64(i+outerBlockSize), float64(dbLen)))

		for k := 0; k < queryLen; k += innerBlockSize {
			kEnd := int(math.Min(float64(k+innerBlockSize), float64(queryLen)))

			for ii := i; ii < iEnd; ii++ {
				d_vec := database[ii]
				dot := scores[ii]

				for kk := k; kk < kEnd-7; kk += 8 {
					dot += query[kk] * d_vec[kk]
					dot += query[kk+1] * d_vec[kk+1]
					dot += query[kk+2] * d_vec[kk+2]
					dot += query[kk+3] * d_vec[kk+3]
					dot += query[kk+4] * d_vec[kk+4]
					dot += query[kk+5] * d_vec[kk+5]
					dot += query[kk+6] * d_vec[kk+6]
					dot += query[kk+7] * d_vec[kk+7]
				}

				for kk := kEnd - (kEnd % 8); kk < kEnd; kk++ {
					dot += query[kk] * d_vec[kk]
				}

				scores[ii] = dot
			}
		}
	}

	return scores
}

func einsum_small(query []float32, database [][]float32) []float32 {
	scores := make([]float32, len(database))
	for i, d_vec := range database {
		var dot float32
		remainder := len(query) % 4

		for j := 0; j < len(query)-remainder; j += 4 {
			dot += query[j] * d_vec[j]
			dot += query[j+1] * d_vec[j+1]
			dot += query[j+2] * d_vec[j+2]
			dot += query[j+3] * d_vec[j+3]
		}

		for j := len(query) - remainder; j < len(query); j++ {
			dot += query[j] * d_vec[j]
		}
		scores[i] = dot
	}
	return scores
}
