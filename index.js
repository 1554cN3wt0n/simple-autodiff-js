const { Tensor } = require("./src/data/tensor");
const { Variable } = require("./src/data/variable");
const { Linear, Sigmoid } = require("./src/nn/layers");

// let x = new Tensor([1, 2], [1, 2]);
// let xv = new Variable(x, null);
// let yv = xv.mul(xv).mul(xv);

// console.log(xv);

// yv.backward(x);

// console.log(xv);
let l = new Linear(2, 4);
let s = new Sigmoid();
let x = new Variable(new Tensor([1, 2], [1, 2]), null);

let o = s.forward(l.forward(x));

console.log(o);
