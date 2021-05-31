package deep

// Neuron is a neural network node
type Neuron struct {
	A     ActivationType `json:"-"`
	In    []*Synapse
	Out   []*Synapse
	Value float32 `json:"-"`
}

// NewNeuron returns a neuron with the given activation
func NewNeuron(activation ActivationType) *Neuron {
	return &Neuron{
		A: activation,
	}
}

func (n *Neuron) fire(training bool) {
	var sum float32
	for _, s := range n.In {
		sum += s.Out
	}
	n.Value = n.Activate(sum, training)

	nVal := n.Value
	for _, s := range n.Out {
		s.fire(nVal)
	}
}

// Activate applies the neurons activation
func (n *Neuron) Activate(x float32, training bool) float32 {
	return GetActivation(n.A).F(x, training)
}

// DActivate applies the derivative of the neurons activation
func (n *Neuron) DActivate(x float32) float32 {
	return GetActivation(n.A).Df(x)
}

// Synapse is an edge between neurons
type Synapse struct {
	Weight  float32
	In, Out float32 `json:"-"`
	IsBias  bool
}

// NewSynapse returns a synapse with the specified initialized weight
func NewSynapse(weight float32) *Synapse {
	return &Synapse{Weight: weight}
}

func (s *Synapse) fire(value float32) {
	s.In = value
	s.Out = s.In * s.Weight
}
