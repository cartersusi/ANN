package ann

import (
	"math"
)

const (
	small_threshold  = 65536   //1 << 16
	medium_threshold = 262144  //1 << 18
	large_threshold  = 1048576 //1 << 20
)

type pair struct {
	val   float32
	index int
}

/*
A.2 Euclidean distance search implementation
@jax.jit
def l2nns(qy, db, db_half_norm):
dots = jax.numpy.einsum(’ik,jk->ij’, qy, db)
dists = db_half_norm - dots
return jax.lax.approx_min_k(dists, k=10, recall_target=0.95)
*/

func Auto[T float32 | float64](qy []float32, db [][]float32, db_half_norm []float32, k int, recall_target ...T) ([]int, []float32) {
	n := len(db)
	d := len(db[0])
	recallTarget := float32(1.00)
	if len(recall_target) > 0 {
		if recall_target[0] > 0.5 {
			recallTarget = float32(recall_target[0])
		}
	}
	if recallTarget > 1.0 {
		recallTarget = 1.0
	}

	nxd := n * d
	if medium_threshold < nxd && nxd < large_threshold {
		recallTarget = 0.95
	} else if nxd >= large_threshold {
		recallTarget = 0.90
	}

	//99% of the time is spent in this function
	dots := Einsum(qy, db, nxd)
	dists := sub(db_half_norm, dots)

	return approx_min_k(dists, k, recallTarget)
}

func approx_min_k(dist []float32, k int, recall_target float32) ([]int, []float32) {
	n := len(dist)
	kf := float64(k)
	L := int(math.Ceil((kf - 1.0) / (1.0 - float64(recall_target))))

	binSize := nextPowerOfTwo(n / L)
	numBins := (n + binSize - 1) / binSize

	binMinValues := make([]float32, numBins)
	binMinIndices := make([]int, numBins)
	for i := range binMinValues {
		binMinValues[i] = float32(math.MaxFloat32)
	}

	for i := 0; i < n; i++ {
		binIdx := i / binSize
		if dist[i] < binMinValues[binIdx] {
			binMinValues[binIdx] = dist[i]
			binMinIndices[binIdx] = i
		}
	}

	indices, values := exact_topK(binMinValues, binMinIndices, k)
	return indices, values
}

func exact_topK(values []float32, indices []int, k int) ([]int, []float32) {
	n := len(values)
	if k > n {
		k = n
	}

	pairs := make([]pair, n)
	for i := 0; i < n; i++ {
		pairs[i] = pair{values[i], indices[i]}
	}

	qs(pairs, 0, n-1, k)
	resultIndices := make([]int, k)
	resultValues := make([]float32, k)
	for i := 0; i < k; i++ {
		resultIndices[i] = pairs[i].index
		resultValues[i] = pairs[i].val
	}

	return resultIndices, resultValues
}

func nextPowerOfTwo(n int) int {
	p := 1
	for p < n {
		p *= 2
	}
	return p
}

func qs(arr []pair, left, right, k int) {
	if left == right {
		return
	}

	pivotIndex := partition(arr, left, right)
	if k-1 == pivotIndex {
		return
	} else if k-1 < pivotIndex {
		qs(arr, left, pivotIndex-1, k)
	} else {
		qs(arr, pivotIndex+1, right, k)
	}
}

func partition(arr []pair, left, right int) int {
	pivot := arr[right].val
	i := left
	for j := left; j < right; j++ {
		if arr[j].val <= pivot {
			arr[i], arr[j] = arr[j], arr[i]
			i++
		}
	}
	arr[i], arr[right] = arr[right], arr[i]
	return i
}

func sub(a, b []float32) []float32 {
	n := len(a)
	c := make([]float32, n)
	for i := 0; i < n; i++ {
		c[i] = a[i] - b[i]
	}
	return c
}
