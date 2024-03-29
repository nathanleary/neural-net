package deep

import math "github.com/chewxy/math32"

// Mode denotes inference mode
type Mode int

const (
	// ModeDefault is unspecified mode
	ModeDefault Mode = 0
	// ModeMultiClass is for one-hot encoded classification, applies softmax output layer
	ModeMultiClass Mode = 1
	// ModeRegression is regression, applies linear output layer
	ModeRegression Mode = 2
	// ModeBinary is binary classification, applies sigmoid output layer
	ModeBinary Mode = 3
	// ModeMultiLabel is for multilabel classification, applies sigmoid output layer
	ModeMultiLabel Mode = 4
)

// OutputActivation returns activation corresponding to prediction mode
func OutputActivation(c Mode) ActivationType {
	switch c {
	case ModeMultiClass:
		return ActivationSoftmax
	case ModeRegression:
		return ActivationLinear
	case ModeBinary, ModeMultiLabel:
		return ActivationSigmoid
	}
	return ActivationNone
}

// GetActivation returns the concrete activation given an ActivationType
func GetActivation(act ActivationType) Differentiable {
	switch act {
	case ActivationSigmoid:
		return Sigmoid{}
	case ActivationTanh:
		return Tanh{}
	case ActivationReLU:
		return ReLU{}
	case ActivationELU:
		return eLU{}
	case ActivationSwish:
		return Swish{}
	case ActivationRootSwish:
		return RootSwish{}
	case ActivationMish:
		return Mish{}
	case ActivationCustom:
		return Custom{}
	case ActivationLinear:
		return Linear{}
	case ActivationSoftmax:
		return Linear{}
	case ActivationDoubleRoot:
		return DoubleRoot{}
	case ActivationRootX:
		return RootX{}
	case ActivationDivX:
		return DivX{}
	case ActivationDoubleDiv:
		return DoubleDiv{}
	case ActivationRootPow:
		return RootPow{}
	case ActivationDoublePow:
		return RootPow{}
	}
	return Linear{}
}

// ActivationType is represents a neuron activation function
type ActivationType int

const (
	// ActivationNone is no activation
	ActivationNone ActivationType = 0
	// ActivationSigmoid is a sigmoid activation
	ActivationSigmoid ActivationType = 1
	// ActivationTanh is hyperbolic activation
	ActivationTanh ActivationType = 2
	// ActivationReLU is rectified linear unit activation
	ActivationReLU ActivationType = 3
	// ActivationLinear is linear activation
	ActivationLinear ActivationType = 4
	// ActivationSoftmax is a softmax activation (per layer)
	ActivationSoftmax ActivationType = 5
	// ActivationELU is a Elu activation
	ActivationELU ActivationType = 6
	// ActivationSwish is a Swish activation
	ActivationSwish ActivationType = 7
	// ActivationMish is a Mish activation
	ActivationMish ActivationType = 8
	// ActivationCustom is a Custom activation
	ActivationCustom ActivationType = 9
	// ActivationCustom is a Custom activation
	ActivationDoubleRoot ActivationType = 10
	// ActivationCustom is a Custom activation
	ActivationRootX ActivationType = 11
	// ActivationMulDiv is a Custom activation
	ActivationDivX ActivationType = 12
	// ActivationMulDiv is a Custom activation
	ActivationDoubleDiv ActivationType = 13
	// ActivationMulDiv is a Custom activation
	ActivationRootPow ActivationType = 14
	// ActivationMulDiv is a Custom activation
	ActivationDoublePow ActivationType = 15
	// ActivationMulDiv is a Custom activation
	ActivationRootSwish ActivationType = 16
)

// Differentiable is an activation function and its first order derivative,
// where the latter is expressed as a function of the former for efficiency
type Differentiable interface {
	F(float32, bool) float32
	Df(float32) float32
}

// Sigmoid is a logistic activator in the special case of a = 1
type Sigmoid struct {
	Mem map[float32]float32
}

// F is Sigmoid(x)
func (a Sigmoid) F(x float32, training bool) float32 { return Logistic(x, 1) }

// Df is Sigmoid'(y), where y = Sigmoid(x)
func (a Sigmoid) Df(y float32) float32 { return y * (1 - y) }

func Sqrt(N float32) float32 {
	return math.Sqrt(N)
}

// DoubleRoot is a logistic activator in the special case of a = 1
type DoubleRoot struct {
	Mem map[float32]float32
}

// F is DoubleRoot(x)
func (a DoubleRoot) F(x float32, training bool) float32 {
	if x == 0 {
		return 0
	} else if x > 0 {
		return Sqrt(x)
	} else {
		return -Sqrt(-x)
	}
}

// Df is DoubleRoot'(y), where y = DoubleRoot(x)
func (a DoubleRoot) Df(x float32) float32 {
	if x == 0 {
		return 0
	} else if x > 0 {
		return 1 / (2 * Sqrt(x))
	} else {
		return 1 / (2 * Sqrt(-x))
	}
}

// RootX is a logistic activator in the special case of a = 1
type DoublePow struct {
	Mem map[float32]float32
}

// F is RootX(x)
func (a DoublePow) F(x float32, training bool) float32 {
	if x == 0 {
		return 0
	} else if x > 0 {
		return x * x
	} else {
		return x * (-x)
	}
}

// Df is DoubleRoot'(y), where y = DoubleRoot(x)
func (a DoublePow) Df(x float32) float32 {
	if x == 0 {
		return 0
	} else if x > 0 {
		return (2 * x)
	} else {
		return -(2 * x)
	}
}

// RootX is a logistic activator in the special case of a = 1
type RootPow struct {
	Mem map[float32]float32
}

// F is RootX(x)
func (a RootPow) F(x float32, training bool) float32 {
	if x == 0 {
		return 0
	} else if x > 0 {
		return ((x + 0.5) * (x + 0.5)) - 0.25
	} else {
		return 0.5 - Sqrt(0.25-x)
	}
}

// Df is DoubleRoot'(y), where y = DoubleRoot(x)
func (a RootPow) Df(x float32) float32 {
	if x == 0 {
		return 0
	} else if x > 0 {
		return (2 * x) + 1
	} else {
		return 1 / (2 * Sqrt(0.25-x))
	}
}

// RootX is a logistic activator in the special case of a = 1
type RootX struct {
	Mem map[float32]float32
}

// F is RootX(x)
func (a RootX) F(x float32, training bool) float32 {
	if x == 0 {
		return 0
	} else if x > 0 {
		return x
	} else {
		return 0.5 - Sqrt(0.25-x)
	}
}

// Df is DoubleRoot'(y), where y = DoubleRoot(x)
func (a RootX) Df(x float32) float32 {
	if x == 0 {
		return 0
	} else if x > 0 {
		return 1
	} else {
		return 1 / (2 * Sqrt(0.25-x))
	}
}

// MulDiv is a logistic activator in the special case of a = 1
type DivX struct {
	Mem map[float32]float32
}

// F is MulDiv(x)
func (a DivX) F(x float32, training bool) float32 {
	if x >= 0 {
		return x
	} else {
		return (1/(x-1) + 1) * -1
	}
}

// Df is MulDiv'(y), where y = MulDiv(x)
func (a DivX) Df(x float32) float32 {
	if x >= 0 {
		return 1
	} else {
		return (1 / ((x - 1) * (x - 1)))
	}
}

// MulDiv is a logistic activator in the special case of a = 1
type DoubleDiv struct {
	Mem map[float32]float32
}

// F is MulDiv(x)
func (a DoubleDiv) F(x float32, training bool) float32 {

	if x == 0 {
		return 0
	} else if x > 0 {
		return (1/(x+1) - 1) * -1
	} else {
		return (1/(x-1) + 1) * -1
	}

}

// Df is MulDiv'(y), where y = MulDiv(x)
func (a DoubleDiv) Df(x float32) float32 {
	if x == 0 {
		return 0
	} else if x > 0 {
		return (1 / ((x + 1) * (x + 1)))
	} else {
		return (1 / ((x - 1) * (x - 1)))
	}
}

// Logistic is the logistic function
func Logistic(x, a float32) float32 {
	return 1 / (1 + math.Exp(-a*x))
}

// Tanh is a hyperbolic activator
type Tanh struct {
	Mem map[float32]float32
}

// F is Tanh(x)
func (a Tanh) F(x float32, training bool) float32 { return (1 - math.Exp(-2*x)) / (1 + math.Exp(-2*x)) }

// Df is Tanh'(y), where y = Tanh(x)
func (a Tanh) Df(y float32) float32 { return 1 - math.Pow(y, 2) }

// ReLU is a rectified linear unit activator
type ReLU struct {
	Mem map[float32]float32
}

// F is ReLU(x)
func (a ReLU) F(x float32, training bool) float32 {

	return math.Max(x, 0)

}

// Df is ReLU'(y), where y = ReLU(x)
func (a ReLU) Df(y float32) float32 {

	if y > 0 {
		return 1
	}
	return 0
}

type eLU struct {
	Mem map[float32]float32
}

// F is ELU(x)
func (a eLU) F(x float32, training bool) float32 {

	if x >= 0 {
		// elu formula
		return x + 0.0000001
	} else {
		return 1.0*math.Pow(math.E, x)*-1 + float32(math.SmallestNonzeroFloat32)
	}

}

// Df is ReLU'(y), where y = ReLU(x)
func (a eLU) Df(y float32) float32 {
	if y > 0 {
		return 1 - 0.0000001
	} else {
		return 1.0*math.Exp(y) - float32(math.SmallestNonzeroFloat32)
	}

}

type Swish struct {
	Mem map[float32]float32
}

// F is Swish(x)
func (a Swish) F(x float32, training bool) float32 {
	// 	if a.Mem == nil {
	// 		a.Mem = map[float32]float32{}
	// 	}
	// 	ans := x * Logistic(x, 1)
	// 	if training {
	// 		a.Mem[ans] = x
	// 	}
	// 	return ans

	return x / (math.Exp(-x) + 1)

}

// Df is swish'(y), where y = Swish(x)
func (a Swish) Df(y float32) float32 {
	// 	x := a.Mem[y]
	// 	delete(a.Mem, y)
	// 	sigX := Logistic(x, 1)
	// 	return y * (sigX * (1 + x*(1-sigX)))
	ey := math.Exp(y)
	ey1 := ey + 1
	return (ey * (ey1 + y)) / (ey1 * ey1)

}

type RootSwish struct {
	Mem map[float32]float32
}

// F is Swish(x)
func (a RootSwish) F(x float32, training bool) float32 {
	// 	if a.Mem == nil {
	// 		a.Mem = map[float32]float32{}
	// 	}
	// 	ans := x * Logistic(x, 1)
	// 	if training {
	// 		a.Mem[ans] = x
	// 	}
	// 	return ans
	if x > 0 {

		return x / (math.Exp(-x) + 1)
	} else {
		return 0.5 - math.Sqrt(0.25-(0.5*x))
	}

}

// Df is swish'(y), where y = Swish(x)
func (a RootSwish) Df(y float32) float32 {
	// 	x := a.Mem[y]
	// 	delete(a.Mem, y)
	// 	sigX := Logistic(x, 1)
	// 	return y * (sigX * (1 + x*(1-sigX)))
	if y > 0 {
		ey := math.Exp(y)
		ey1 := ey + 1
		return (ey * (ey1 + y)) / (ey1 * ey1)
	} else {
		return 1 / (2 * math.Sqrt(1-(2*y)))
	}

}

type Mish struct {
	Mem map[float32]float32
}

// F is Mish(x)
func (a Mish) F(x float32, training bool) float32 {
	if a.Mem == nil {
		a.Mem = map[float32]float32{}
	}

	ans := x * math.Tanh(math.Log(1+math.Exp(x)))
	if training {
		a.Mem[ans] = x
	}
	return ans

}

// Df is Mish'(y), where y = Mish(x)
func (a Mish) Df(y float32) float32 {
	x := a.Mem[y]
	delete(a.Mem, y)
	sigX := Logistic(x, 1)
	xTanhSp := math.Tanh(math.Log(1 + math.Exp(x)))
	return y * (xTanhSp + x*sigX*(1-xTanhSp*xTanhSp))

}

type Custom struct {
	Mem map[float32]float32
}

var customF func(float32) float32
var customDf func(float32, float32) float32

func SetCustomF(F func(float32) float32) {
	customF = F
}

func SetCustomDf(Df func(float32, float32) float32) {
	customDf = Df
}

// F is Custom(x)
func (a Custom) F(x float32, training bool) float32 {

	if a.Mem == nil {
		a.Mem = map[float32]float32{}
	}

	if customF != nil {
		ans := customF(x)
		if training {
			a.Mem[ans] = x
		}
		return ans
	} else {
		ans := x
		if training {
			a.Mem[ans] = x
		}
		return x
	}

}

// Df is Custom'(y), where y = Custom(x)
func (a Custom) Df(y float32) float32 {
	x := a.Mem[y]
	delete(a.Mem, y)

	if customDf != nil {
		return customDf(y, x)
	} else {
		return x
	}

}

// Linear is a linear activator
type Linear struct {
	Mem map[float32]float32
}

// F is the identity function
func (a Linear) F(x float32, training bool) float32 { return x }

// Df is constant
func (a Linear) Df(x float32) float32 { return 1 }
