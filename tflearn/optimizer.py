# -*- coding: UTF-8 -*-
#定义目标函数， loss=(x−3)2， 求goal最小时，x的值：
import tensorflow as tf
x = tf.Variable(tf.truncated_normal([1]), name="x")
goal = tf.pow(x-3,2, name="goal")
with tf.Session() as sess:
    x.initializer.run()
    print x.eval()
    print goal.eval()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_step = optimizer.minimize(goal)

def train():
    with tf.Session() as sess:
        x.initializer.run()
        for i in range(100):
            print "x: ", x.eval()
            train_step.run()
            print "goal: ",goal.eval()
train()
