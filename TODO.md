# TODO List

* [ ] auto reshaping data
* [ ] Network class

  ```python
  Network(
    [a := Linear(3, 10), Tanh, b := Linear(10)],
    [b, Tanh, [a, c := Linear(4)]],
    [[d := Linear(3, 4), Randn(4)], Mul, e := Add],
    [c, e, f := Softmax]
  )
  ```

  This should build a network like this:

  ```txt
    🡧⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺🡤  
  a -> Tanh -> b -> Tanh -> c
                             ╲
  d ⎺⎺⎺⎺⎺⎺⎺⎺🡦                 🡦
            Mul ─────────────🡢 e(Add) 🡢 f(Softmax)
  Randn(4) 🡥
  ```

  Each list is a chain of connected nodes. A nested list connects all its items
  with its predecessor and successor.

* [ ] implement CNN
* [ ] implement RNN
* [ ] implement VAE
* [ ] add regularization
* [ ] implement LSTM ?
  * [ ] train an RNN to generate text imitating 史记
* [ ] implement GAN ?


Update Function:
I should implement two things for a Function: its gradient of output wrt. input,
and its gradient of output wrt. weights.
