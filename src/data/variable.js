const { Tensor } = require("./tensor");

const {
  AddBackward,
  SubBackward,
  MulBackward,
  DivBackward,
  DotBackward,
} = require("./backward");

class Variable {
  constructor(t, backward_hook) {
    this.val = t;
    if (this.backward_hook) {
      this.grad = null;
    } else {
      this.grad = new Tensor(Array(t.data.length), [...t.shape]);
    }
    this.grad.zeros();
    this.backward_hook = backward_hook; // if this is null it means it is a leaf of the graph
  }
  backward(loss) {
    if (this.backward_hook) {
      this.backward_hook.call(loss);
    } else {
      this.grad = this.grad.add(loss);
    }
  }
  add(v) {
    let new_val = this.val.add(v.val);
    return new Variable(new_val, new AddBackward(this, v));
  }
  sub(v) {
    let new_val = this.val.sub(v.val);
    return new Variable(new_val, new SubBackward(this, v));
  }
  mul(v) {
    let new_val = this.val.mul(v.val);
    return new Variable(new_val, new MulBackward(this, v));
  }
  div(v) {
    let new_val = this.val.div(v.val);
    return new Variable(new_val, new DivBackward(this, v));
  }
  dot(v) {
    let new_val = this.val.dot(v.val);
    return new Variable(new_val, new DotBackward(this, v));
  }
}

module.exports = { Variable };
