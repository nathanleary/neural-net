package training

import (
	"sync"
	"time"

	deep "github.com/nathanleary/neural-net"
)

// BatchTrainer implements parallelized batch training
type BatchTrainer struct {
	*internalb
	verbosity   int
	batchSize   int
	parallelism int
	solver      Solver
	printer     *StatsPrinter
}

type internalb struct {
	deltas            [][][]float32
	partialDeltas     [][][][]float32
	accumulatedDeltas [][][]float32
	moments           [][][]float32
}

func newBatchTraining(layers []*deep.Layer, parallelism int) *internalb {
	deltas := make([][][]float32, parallelism)
	partialDeltas := make([][][][]float32, parallelism)
	accumulatedDeltas := make([][][]float32, len(layers))
	for w := 0; w < parallelism; w++ {
		deltas[w] = make([][]float32, len(layers))
		partialDeltas[w] = make([][][]float32, len(layers))

		for i, l := range layers {
			deltas[w][i] = make([]float32, len(l.Neurons))
			accumulatedDeltas[i] = make([][]float32, len(l.Neurons))
			partialDeltas[w][i] = make([][]float32, len(l.Neurons))
			for j, n := range l.Neurons {
				partialDeltas[w][i][j] = make([]float32, len(n.In))
				accumulatedDeltas[i][j] = make([]float32, len(n.In))
			}
		}
	}
	return &internalb{
		deltas:            deltas,
		partialDeltas:     partialDeltas,
		accumulatedDeltas: accumulatedDeltas,
	}
}

// NewBatchTrainer returns a BatchTrainer
func NewBatchTrainer(solver Solver, verbosity, batchSize, parallelism int) *BatchTrainer {
	return &BatchTrainer{
		solver:      solver,
		verbosity:   verbosity,
		batchSize:   iparam(batchSize, 1),
		parallelism: iparam(parallelism, 1),
		printer:     NewStatsPrinter(),
	}
}

// Train trains n
func (t *BatchTrainer) Train(n *deep.Neural, examples, validation Examples, iterations int) {
	t.internalb = newBatchTraining(n.Layers, t.parallelism)

	train := make(Examples, len(examples))
	copy(train, examples)

	workCh := make(chan Example, t.parallelism)
	nets := make([]*deep.Neural, t.parallelism)

	wg := sync.WaitGroup{}
	for i := 0; i < t.parallelism; i++ {
		nets[i] = deep.NewNeural(n.Config)

		go func(id int, workCh <-chan Example) {
			n := nets[id]
			for e := range workCh {
				n.Forward(e.Input, true)
				t.calculateDeltas(n, e.Response, id)
				wg.Done()
			}
		}(i, workCh)
	}

	t.printer.Init(n)
	t.solver.Init(n.NumWeights())

	ts := time.Now()
	for it := 1; it <= iterations; it++ {
		train.Shuffle()
		batches := train.SplitSize(t.batchSize)

		for _, b := range batches {
			currentWeights := n.Weights()
			for _, n := range nets {
				n.ApplyWeights(currentWeights)
			}

			wg.Add(len(b))
			for _, item := range b {
				workCh <- item
			}
			wg.Wait()
			wg.Add(len(t.partialDeltas))
			for _, wPD := range t.partialDeltas {

				go func(wPD [][][]float32) {
					defer wg.Done()
					for i, iPD := range wPD {
						iAD := t.accumulatedDeltas[i]
						for j, jPD := range iPD {
							jAD := iAD[j]
							for k, v := range jPD {
								jAD[k] += v
								jPD[k] = 0
							}
						}
					}
				}(wPD)
			}

			wg.Wait()

			t.update(n, it)
		}

		if t.verbosity > 0 && it%t.verbosity == 0 && len(validation) > 0 {
			t.printer.PrintProgress(n, validation, time.Since(ts), it)
		}
	}
}

func (t *BatchTrainer) calculateDeltas(n *deep.Neural, ideal []float32, wid int) {
	loss := deep.GetLoss(n.Config.Loss)
	deltas := t.deltas[wid]
	partialDeltas := t.partialDeltas[wid]
	lastDeltas := deltas[len(n.Layers)-1]

	for i, n := range n.Layers[len(n.Layers)-1].Neurons {
		lastDeltas[i] = loss.Df(
			n.Value,
			ideal[i],
			n.DActivate(n.Value))
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {

		l := n.Layers[i]
		iD := deltas[i]
		nextD := deltas[i+1]

		for j, n := range l.Neurons {
			var sum float32
			for k, s := range n.Out {
				sum += s.Weight * nextD[k]
			}
			iD[j] = n.DActivate(n.Value) * sum
		}

	}

	for i, l := range n.Layers {

		iD := deltas[i]
		iPD := partialDeltas[i]
		for j, n := range l.Neurons {
			jD := iD[j]
			jPD := iPD[j]
			for k, s := range n.In {
				jPD[k] += jD * s.In
			}
		}

	}

}

func (t *BatchTrainer) update(n *deep.Neural, it int) {
	// var idx int

	mut := sync.Mutex{}
	ch := make(chan bool, len(n.Layers))
	for i, l := range n.Layers {
		go func(l *deep.Layer, i int) {
			idx := i
			iAD := t.accumulatedDeltas[i]
			for j, n := range l.Neurons {
				jAD := iAD[j]
				for k, s := range n.In {

					update := t.solver.Update(s.Weight,
						jAD[k],
						it,
						idx)

					mut.Lock()
					s.Weight += update
					mut.Unlock()

					jAD[k] = 0

					idx++
				}
			}

			ch <- false
		}(l, i)
	}

	for <-ch {

	}
}
