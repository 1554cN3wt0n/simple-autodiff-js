const { Variable } = require("../data/variable");
const { _sigmoid } = require("./utils");
const { NopBackward, SigmoidBackward } = require("../data/backward");

function sigmoid(v) {
  let new_val = v.val.apply_unitary_op(_sigmoid);
  let o = new Variable(new_val, new NopBackward());
  o.backward_hook = new SigmoidBackward(v, o);
  return o;
}

module.exports = { sigmoid };
