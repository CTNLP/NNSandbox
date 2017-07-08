import tensorflow as tf;

3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

node1 = tf.constant(3.0,dtype=tf.float32);
node2 = tf.constant(4.0); # also tf.float32 implicitly

print(node1,node2);

sess = tf.Session(); # session is necessary to actually run/evaluate a computation graph
print(sess.run([node1,node2]));

node3 = tf.add(node1,node2); # simple addition of constants
print("node3: ",node3);
print("evaluates to: ",sess.run(node3));

a = tf.placeholder(tf.float32); # open space into which a float32 value or array can be inserted later
b = tf.placeholder(tf.float32);
addition_node = a + b; # + is a method of a which can be used instead of tf.add(a,b);

print(sess.run(addition_node, {a: 5, b : 2}));
print(sess.run(addition_node, {a: [5,4], b : [2,1]}));

add_and_triple = addition_node * 3. # dot indicates a float, * is another alias
print(sess.run(add_and_triple, {a: [3,1], b:4.5})) # addition is well defined when adding scalars and vectors, same for
# multiplication

W = tf.Variable([.3],dtype=tf.float32); # variable that is trainable with float32 type and initial value .3 in a 1
# dimensional vector
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32) # this is where the actual values will be plugged in
linear_model = W * x + b

# !! variables must be initialized before their first use !!
init = tf.global_variables_initializer()
sess.run(init)

# evaluate current state of model using inputs 1,2,3,4, they will each be multiplied with W and then b will be added,
# producing a vector of responses
print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32) # will be used as a response value
squared_deltas = tf.square(linear_model - y) # computing squared loss
loss = tf.reduce_sum(squared_deltas) # this takes the vector of responses and sums them up
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# variables are changed using operations such as tensorflow.assign
# note that doing this requires another call to the session
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# now we pick suboptimal values again, so that we can then  find the correct ones by optimization
sess.run(init);

# creates a simple optimizer based on gradient descent, with step size 0.01
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# repeatedly train on the dataset, each time that train is run, one optimization step is taken
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print("training results: ");
print(sess.run([W, b]))
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

