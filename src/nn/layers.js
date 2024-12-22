const { Tensor } = require("../data/tensor");
const { Variable } = require("../data/variable");
const { sigmoid } = require("../fn/act_fun");

class Linear {
  constructor(n_input, n_output) {
    let W_ = new Tensor(Array(n_input * n_output), [n_input, n_output]);
    let b_ = new Tensor(Array(n_output), [1, n_output]);
    W_.random();
    b_.random();
    this.W = new Variable(W_, null);
    this.b = new Variable(b_, null);
  }

  forward(x) {
    return x.dot(this.W).add(this.b);
  }
}

class Sigmoid {
  constructor() {}

  forward(x) {
    return sigmoid(x);
  }
}

module.exports = {
  Linear,
  Sigmoid,
};
