package deep

import math "github.com/chewxy/math32"

// Mean of xx
func Mean(xx []float32) float32 {
	var sum float32
	for _, x := range xx {
		sum += x
	}
	return sum / float32(len(xx))
}

// Variance of xx
func Variance(xx []float32) float32 {
	if len(xx) == 1 {
		return 0.0
	}
	m := Mean(xx)

	var variance float32
	for _, x := range xx {
		variance += math.Pow((x - m), 2)
	}

	return variance / float32(len(xx)-1)
}

// StandardDeviation of xx
func StandardDeviation(xx []float32) float32 {
	return math.Sqrt(Variance(xx))
}

// Standardize (z-score) shifts distribution to μ=0 σ=1
func Standardize(xx []float32) {
	m := Mean(xx)
	s := StandardDeviation(xx)

	if s == 0 {
		s = 1
	}

	for i, x := range xx {
		xx[i] = (x - m) / s
	}
}

// Normalize scales to (0,1)
func Normalize(xx []float32) {
	min, max := Min(xx), Max(xx)
	for i, x := range xx {
		xx[i] = (x - min) / (max - min)
	}
}

// Min is the smallest element
func Min(xx []float32) float32 {
	min := xx[0]
	for _, x := range xx {
		if x < min {
			min = x
		}
	}
	return min
}

// Max is the largest element
func Max(xx []float32) float32 {
	max := xx[0]
	for _, x := range xx {
		if x > max {
			max = x
		}
	}
	return max
}

// ArgMax is the index of the largest element
func ArgMax(xx []float32) int {
	max, idx := xx[0], 0
	for i, x := range xx {
		if x > max {
			max, idx = xx[i], i
		}
	}
	return idx
}

// Sgn is signum
func Sgn(x float32) float32 {
	switch {
	case x < 0:
		return -1.0
	case x > 0:
		return 1.0
	}
	return 0
}

// Sum is sum
func Sum(xx []float32) (sum float32) {
	for _, x := range xx {
		sum += x
	}
	return
}

// Softmax is the softmax function
func Softmax(xx []float32) []float32 {
	out := make([]float32, len(xx))
	var sum float32
	max := Max(xx)
	for i, x := range xx {
		out[i] = math.Exp(x - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Round to nearest integer
func Round(x float32) float32 {
	return math.Floor(x + .5)
}

// Dot product
func Dot(xx, yy []float32) float32 {
	var p float32
	for i := range xx {
		p += xx[i] * yy[i]
	}
	return p
}
