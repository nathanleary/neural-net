package training

import math "github.com/chewxy/math32"

// Solver implements an update rule for training a NN
type Solver interface {
	Init(size int)
	Update(value, gradient float32, iteration, idx int) float32
}

// SGD is stochastic gradient descent with nesterov/momentum
type SGD struct {
	lr       float32
	decay    float32
	momentum float32
	nesterov bool
	moments  []float32
}

// NewSGD returns a new SGD solver
func NewSGD(lr, momentum, decay float32, nesterov bool) *SGD {
	return &SGD{
		lr:       fparam(lr, 0.01),
		decay:    decay,
		momentum: momentum,
		nesterov: nesterov,
	}
}

// Init initializes vectors using number of weights in network
func (o *SGD) Init(size int) {
	o.moments = make([]float32, size)
}

// Update returns the update for a given weight
func (o *SGD) Update(value, gradient float32, iteration, idx int) float32 {
	lr := o.lr / (1 + o.decay*float32(iteration))

	o.moments[idx] = o.momentum*o.moments[idx] - lr*gradient

	if o.nesterov {
		o.moments[idx] = o.momentum*o.moments[idx] - lr*gradient
	}

	return o.moments[idx]
}

// Adam is an Adam solver
type Adam struct {
	lr      float32
	beta    float32
	beta2   float32
	epsilon float32

	v, m []float32
}

// NewAdam returns a new Adam solver
func NewAdam(lr, beta, beta2, epsilon float32) *Adam {
	return &Adam{
		lr:      fparam(lr, 0.001),
		beta:    fparam(beta, 0.9),
		beta2:   fparam(beta2, 0.999),
		epsilon: fparam(epsilon, 1e-8),
	}
}

// Init initializes vectors using number of weights in network
func (o *Adam) Init(size int) {
	o.v, o.m = make([]float32, size), make([]float32, size)
}

// Update returns the update for a given weight
func (o *Adam) Update(value, gradient float32, t, idx int) float32 {
	lrt := o.lr * (math.Sqrt(1.0 - math.Pow(o.beta2, float32(t)))) /
		(1.0 - math.Pow(o.beta, float32(t)))
	o.m[idx] = o.beta*o.m[idx] + (1.0-o.beta)*gradient
	o.v[idx] = o.beta2*o.v[idx] + (1.0-o.beta2)*math.Pow(gradient, 2.0)

	return -lrt * (o.m[idx] / (math.Sqrt(o.v[idx]) + o.epsilon))
}

func fparam(val, fallback float32) float32 {
	if val == 0.0 {
		return fallback
	}
	return val
}

func iparam(val, fallback int) int {
	if val == 0 {
		return fallback
	}
	return val
}
