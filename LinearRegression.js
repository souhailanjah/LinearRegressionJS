//solve the equation y = m*x + b
let train_xs = [];
let train_ys = [];

let m, b;
//create optimizer with learningRate = 0.5
const optimizer = tf.train.sgd(0.5);

function setup() {
  createCanvas(400, 400);
  //set rand m and b [0,1]
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

//loss = (prediction - y)²
function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

//build the model
function predict(x) {
  //transofrm the x [] into a tensor
  const xs = tf.tensor1d(x);
  // y = mx + b;
  const ys = xs.mul(m).add(b);
  return ys;
}

//fill input [] with x and y from onClick
function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
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
    let px = map(train_xs[i], 0, 1, 0, width);
    let py = map(train_ys[i], 0, 1, height, 0);
    point(px, py);
  }


  //draw a line using (x1,y1) and (x2,y2)
  const lineX = [0, 1];

  const ys = tf.tidy(() => predict(lineX));
  let lineY = ys.dataSync();
  //draw the line and dispose of ys
  ys.dispose();

  //value of x is between 0 and 1 which is scaled between 0 and width
  let x1 = map(lineX[0], 0, 1, 0, width);
  let x2 = map(lineX[1], 0, 1, 0, width);

  //value of y is between 0 and 1 which is scaled between 0 and height
  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);

  strokeWeight(2);
  line(x1, y1, x2, y2);

}