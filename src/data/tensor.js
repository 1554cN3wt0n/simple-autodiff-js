class Tensor {
  constructor(data, shape) {
    this.data = data;
    this.shape = shape;
    this.stride = Array(this.shape.length);
    let acc = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      this.stride[i] = acc;
      acc *= this.shape[i];
    }
  }

  match_dimension(m) {
    if (this.shape.length != m.shape.length) {
      return false;
    }
    for (let i = 0; i < this.shape.length; i++) {
      if (m.stride[i] != 0 && this.shape[i] != m.shape[i]) {
        return false;
      }
    }
    return true;
  }

  broadcast(i) {
    if (this.shape[i] == 1) {
      this.stride[i] = 0;
    }
  }

  reshape(new_shape) {
    this.shape = new_shape;
    this.stride = Array(this.shape.length);
    let acc = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      this.stride[i] = acc;
      acc *= this.shape[i];
    }
  }

  set_stride(new_stride) {
    this.stride = new_stride;
  }

  random() {
    for (let i = 0; i < this.data.length; i++) {
      this.data[i] = Math.random();
    }
  }

  zeros() {
    for (let i = 0; i < this.data.length; i++) {
      this.data[i] = 0;
    }
  }

  ones() {
    for (let i = 0; i < this.data.length; i++) {
      this.data[i] = 1;
    }
  }

  at(idx) {
    let s = 0;
    for (let i = 0; i < idx.length; i++) {
      s += idx[i] * this.stride[i];
    }
    return this.data[s];
  }

  set(idx, val) {
    let s = 0;
    for (let i = 0; i < idx.length; i++) {
      s += idx[i] * this.stride[i];
    }
    this.data[s] = val;
  }

  get_idx(i) {
    let idx = [];
    for (let j = 0; j < this.stride.length; j++) {
      let t_ = Math.floor(i / this.stride[j]);
      i = i - t_ * this.stride[j];
      idx.push(t_);
    }
    return idx;
  }

  apply_unitary_op(f) {
    let new_data = [];
    for (let i = 0; i < this.data.length; i++) {
      new_data.push(f(this.data[i]));
    }
    return new Tensor(new_data, [...this.shape]);
  }

  apply_binary_op(m, f) {
    if (!this.match_dimension(m)) {
      return null;
    }
    let new_data = [];
    for (let i = 0; i < this.data.length; i++) {
      let idx = this.get_idx(i);
      new_data.push(f(this.data[i], m.at(idx)));
    }
    return new Tensor(new_data, [...this.shape]);
  }

  add(m) {
    if (!this.match_dimension(m)) {
      return null;
    }
    let new_data = [];
    for (let i = 0; i < this.data.length; i++) {
      let idx = this.get_idx(i);
      new_data.push(this.data[i] + m.at(idx));
    }
    return new Tensor(new_data, [...this.shape]);
  }

  sub(m) {
    if (!this.match_dimension(m)) {
      return null;
    }
    let new_data = [];
    for (let i = 0; i < this.data.length; i++) {
      let idx = this.get_idx(i);
      new_data.push(this.data[i] - m.at(idx));
    }
    return new Tensor(new_data, [...this.shape]);
  }

  mul(m) {
    if (!this.match_dimension(m)) {
      return null;
    }
    let new_data = [];
    for (let i = 0; i < this.data.length; i++) {
      let idx = this.get_idx(i);
      new_data.push(this.data[i] * m.at(idx));
    }
    return new Tensor(new_data, [...this.shape]);
  }

  div(m) {
    if (!this.match_dimension(m)) {
      return null;
    }
    let new_data = [];
    for (let i = 0; i < this.data.length; i++) {
      let idx = this.get_idx(i);
      new_data.push(this.data[i] / m.at(idx));
    }
    return new Tensor(new_data, [...this.shape]);
  }

  times(s) {
    let new_data = [];
    for (let i = 0; i < this.data.length; i++) {
      new_data.push(this.data[i] * s);
    }
    return new Tensor(new_data, [...this.shape]);
  }

  dot(m) {
    // TO IMPROVE
    let o = new Tensor(Array(this.shape[0] * m.shape[1]), [
      this.shape[0],
      m.shape[1],
    ]);
    for (let i = 0; i < this.shape[0]; i++) {
      for (let j = 0; j < m.shape[1]; j++) {
        let s = 0;
        for (let k = 0; k < this.shape[1]; k++) {
          s += this.at([i, k]) * m.at([k, j]);
        }
        o.set([i, j], s);
      }
    }
    return o;
  }

  transpose(i, j) {
    let o = new Tensor(this.data, [...this.shape]);

    let x_ = o.stride[i];
    o.stride[i] = o.stride[j];
    o.stride[j] = x_;

    x_ = o.shape[i];
    o.shape[i] = o.shape[j];
    o.shape[j] = x_;

    return o;
  }
}

module.exports = { Tensor };
