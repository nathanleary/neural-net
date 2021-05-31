package deep

import "math/rand"

// A WeightInitializer returns a (random) weight
type WeightInitializer func() float32

// NewUniform returns a uniform weight generator
func NewUniform(stdDev, mean float32) WeightInitializer {
	return func() float32 { return Uniform(stdDev, mean) }
}

// Uniform samples a value from u(mean-stdDev/2,mean+stdDev/2)
func Uniform(stdDev, mean float32) float32 {
	return (rand.Float32()-0.5)*stdDev + mean

}

// NewNormal returns a normal weight generator
func NewNormal(stdDev, mean float32) WeightInitializer {
	return func() float32 { return Normal(stdDev, mean) }
}

// Normal samples a value from N(μ, σ)
func Normal(stdDev, mean float32) float32 {
	return float32(rand.NormFloat64())*stdDev + mean
}
