let train_xs = [];
let train_ys = [];

let a, b, c, d;
//create optimizer with learningRate = 0.5
const optimizer = tf.train.adam(0.2);

function setup() {
  createCanvas(400, 400);
  //set rand m and b [0,1]
  a = tf.variable(tf.scalar(random(-1,1)));
  b = tf.variable(tf.scalar(random(-1,1)));
  c = tf.variable(tf.scalar(random(-1,1)));
  d = tf.variable(tf.scalar(random(-1,1)));
  
}

//loss = (prediction - y)²
function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

//build the model
function predict(x) {
  //transofrm the x [] into a tensor
  const xs = tf.tensor1d(x);
  // y = ax^3 + bx^2 + cx + d
  const ys = xs.pow(tf.scalar(3)).mul(a)
    .add(xs.square().mul(b))
    .add(xs.mul(c))
    .add(d);
  return ys;
}

//fill input [] with x and y from onClick
function mousePressed() {
  let x = map(mouseX, 0, width, -1, 1);
  let y = map(mouseY, 0, height, 1, -1);
  train_xs.push(x);
  train_ys.push(y);
}

function draw() {

  tf.tidy(() => {
    if (train_xs.length > 0) { //initially [] are empty
        //transform the y [] into a tensor 
      const ys = tf.tensor1d(train_ys);
        // use optimizer to minimize loss based on the prediction from the training input and the results got in actuals ys
      optimizer.minimize(() => loss(predict(train_xs), ys));
    }
  });

  background(0);

  //mark the strokes onClick
  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < train_xs.length; i++) {
    let px = map(train_xs[i], -1, 1, 0, width);
    let py = map(train_ys[i], -1, 1, height, 0);
    point(px, py);
  }
  //get array of x and y to draw the shape
  const curveX = [];
  for(let x = -1 ; x <= 1; x+=0.05){
    curveX.push(x);
  }

  const lineX = [-1, 1];

  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();
  
  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for (let  i = 0; i < curveX.length ; i++){
    let x  = map(curveX[i],-1,1,0,width)
    let y  = map(curveY[i],-1,1,height,0)
    vertex(x,y);
  }
  endShape();

}