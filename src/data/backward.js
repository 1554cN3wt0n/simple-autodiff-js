class NopBackward {
  constructor() {}
  call(loss) {}
}

class AddBackward {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
  call(loss) {
    this.x.backward(loss);
    this.y.backward(loss);
  }
}

class SubBackward {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
  call(loss) {
    this.x.backward(loss);
    this.y.backward(loss.times(-1));
  }
}

class MulBackward {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
  call(loss) {
    this.x.backward(loss.mul(this.y.val));
    this.y.backward(loss.mul(this.x.val));
  }
}

class DivBackward {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
  helper(a, b) {
    return -a / b ** 2;
  }
  call(loss) {
    this.x.backward(loss.div(this.y.val));
    this.y.backward(
      loss.mul(this.x.val.apply_binary_op(this.y.val, this.helper))
    );
  }
}

class DotBackward {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
  call(loss) {
    this.x.backward(loss.dot(this.y.transpose(0, 1)));
    this.y.backward(this.x.transpose(0, 1).dot(loss));
  }
}

class SigmoidBackward {
  constructor(x, o) {
    this.x = x;
    this.o = o;
  }
  helper(x) {
    return x * (1 - x);
  }
  call(loss) {
    this.x.backward(loss.mul(this.o.val.apply_unitary_op(this.helper)));
  }
}

module.exports = {
  NopBackward,
  AddBackward,
  SubBackward,
  MulBackward,
  DivBackward,
  DotBackward,
  SigmoidBackward,
};
