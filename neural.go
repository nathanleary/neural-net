package deep

import (
	"fmt"
)

// Neural is a neural network
type Neural struct {
	Shift        []float32
	Significance []float32
	Layers       []*Layer
	Biases       [][]*Synapse
	Config       *Config
}

// Config defines the network topology, activations, losses etc
type Config struct {

	// Number of inputs
	Inputs int
	// Defines topology:
	// For instance, [5 3 3] signifies a network with two hidden layers
	// containing 5 and 3 nodes respectively, followed an output layer
	// containing 3 nodes.
	Layout []int
	// Activation functions: {ActivationTanh, ActivationReLU, ActivationSigmoid}
	Activation []ActivationType
	// Solver modes: {ModeRegression, ModeBinary, ModeMultiClass, ModeMultiLabel}
	Mode Mode
	// Initializer for weights: {NewNormal(σ, μ), NewUniform(σ, μ)}
	Weight WeightInitializer `json:"-"`
	// Loss functions: {LossCrossEntropy, LossBinaryCrossEntropy, LossMeanSquared}
	Loss LossType
	// Apply bias nodes
	Bias bool
	// this is a training variable to help decide on the amount of significance any input variable has when training
	Significance float32
	// this is a training variable that adds a constant number to each input variable to shift the number up or down
	Shift float32
}

// NewNeural returns a new neural network
func NewNeural(c *Config) *Neural {

	if c.Weight == nil {
		c.Weight = NewUniform(0.5, 0)
	}
	// if c.Activation == ActivationNone {
	// taking this out...
	// 	c.Activation = ActivationSigmoid
	// }
	if c.Loss == LossNone {
		switch c.Mode {
		case ModeMultiClass, ModeMultiLabel:
			c.Loss = LossCrossEntropy
		case ModeBinary:
			c.Loss = LossBinaryCrossEntropy
		default:
			c.Loss = LossMeanSquared
		}
	}

	layers := initializeLayers(c)

	var biases [][]*Synapse
	if c.Bias {
		biases = make([][]*Synapse, len(layers))
		for i := 0; i < len(layers); i++ {
			if c.Mode == ModeRegression && i == len(layers)-1 {
				continue
			}
			biases[i] = layers[i].ApplyBias(c.Weight)
		}
	}

	significance := make([]float32, c.Inputs)
	shift := make([]float32, c.Inputs)

	for i, _ := range significance {
		significance[i] = 1.0
		shift[i] = 0.0
	}

	return &Neural{
		Shift:        shift,
		Significance: significance,
		Layers:       layers,
		Biases:       biases,
		Config:       c,
	}
}

func initializeLayers(c *Config) []*Layer {
	layers := make([]*Layer, len(c.Layout))
	for i := range layers {
		act := ActivationLinear
		if i == (len(layers)-1) && c.Mode != ModeDefault {
			act = OutputActivation(c.Mode)
		} else {
			act = c.Activation[i]
		}
		layers[i] = NewLayer(c.Layout[i], act)
	}

	for i := 0; i < len(layers)-1; i++ {
		layers[i].Connect(layers[i+1], c.Weight)
	}

	for _, neuron := range layers[0].Neurons {
		neuron.In = make([]*Synapse, c.Inputs)
		for i := range neuron.In {
			neuron.In[i] = NewSynapse(c.Weight())
		}
	}

	return layers
}

func (n *Neural) fire(training bool) {

	for _, b := range n.Biases {

		for _, s := range b {
			s.fire(1)

		}

	}

	for _, l := range n.Layers {

		l.fire(training)

	}

}

// Forward computes a forward pass
func (n *Neural) Forward(input []float32, training bool) error {
	if len(input) != n.Config.Inputs {
		return fmt.Errorf("Invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}

	for _, nrn := range n.Layers[0].Neurons {

		for i := 0; i < len(input); i++ {

			nrn.In[i].fire((input[i] + n.Shift[i]) * n.Significance[i])

		}

	}

	n.fire(training)
	return nil
}

// Predict computes a forward pass and returns a prediction
func (n *Neural) Predict(input []float32) []float32 {

	n.Forward(input, false)

	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]float32, len(outLayer.Neurons))
	for i, neuron := range outLayer.Neurons {
		out[i] = neuron.Value
	}
	return out
}

// NumWeights returns the number of weights in the network
func (n *Neural) NumWeights() (num int) {
	for _, l := range n.Layers {
		for _, n := range l.Neurons {
			num += len(n.In)
		}
	}
	return
}

func (n *Neural) String() string {
	var s string
	for _, l := range n.Layers {
		s = fmt.Sprintf("%s\n%s", s, l)
	}
	return s
}
