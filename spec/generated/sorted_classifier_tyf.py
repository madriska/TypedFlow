
import tensorflow as tf
def mkModel(optimizer=tf.train.AdamOptimizer()):
  
  var0=tf.placeholder(tf.bool, shape=[], name="training_phase")
  var1=tf.placeholder(tf.float32, shape=[100,5], name="x")
  var2=tf.placeholder(tf.float32, shape=[100,2], name="y")
  var5=tf.random_uniform([5,10], minval=-0.6324555, maxval=0.6324555, dtype=tf.float32) # 3
  assert var5.shape.as_list() == [5,10]
  var6=tf.Variable(var5, name="w1_w", trainable=True)
  var7=tf.truncated_normal([10], stddev=0.1, dtype=tf.float32) # 4
  assert var7.shape.as_list() == [10]
  var8=tf.Variable(var7, name="w1_bias", trainable=True)
  var11=tf.random_uniform(
          [10,10], minval=-0.5477226, maxval=0.5477226, dtype=tf.float32) # 9
  assert var11.shape.as_list() == [10,10]
  var12=tf.Variable(var11, name="w2_w", trainable=True)
  var13=tf.truncated_normal([10], stddev=0.1, dtype=tf.float32) # 10
  assert var13.shape.as_list() == [10]
  var14=tf.Variable(var13, name="w2_bias", trainable=True)
  var17=tf.random_uniform(
          [10,2], minval=-0.70710677, maxval=0.70710677, dtype=tf.float32) # 15
  assert var17.shape.as_list() == [10,2]
  var18=tf.Variable(var17, name="w3_w", trainable=True)
  var19=tf.truncated_normal([2], stddev=0.1, dtype=tf.float32) # 16
  assert var19.shape.as_list() == [2]
  var20=tf.Variable(var19, name="w3_bias", trainable=True)
  var21=var2
  assert var21.shape.as_list() == [100,2]
  var22=var1
  assert var22.shape.as_list() == [100,5]
  var23=tf.reshape(var22, [100,5])
  assert var23.shape.as_list() == [100,5]
  var24=var6
  assert var24.shape.as_list() == [5,10]
  var25=tf.matmul(var23, var24)
  assert var25.shape.as_list() == [100,10]
  var26=tf.reshape(var25, [100,10])
  assert var26.shape.as_list() == [100,10]
  var27=var8
  assert var27.shape.as_list() == [10]
  var28=tf.broadcast_to(tf.reshape(var27, [1,10]), [100,10])
  assert var28.shape.as_list() == [100,10]
  var29=tf.add(var26, var28)
  assert var29.shape.as_list() == [100,10]
  var30=tf.keras.layers.BatchNormalization(axis=1)(var29)
  assert var30.shape.as_list() == [100,10]
  var31=tf.nn.relu(var30)
  assert var31.shape.as_list() == [100,10]
  var32=tf.reshape(var31, [100,10])
  assert var32.shape.as_list() == [100,10]
  var33=var12
  assert var33.shape.as_list() == [10,10]
  var34=tf.matmul(var32, var33)
  assert var34.shape.as_list() == [100,10]
  var35=tf.reshape(var34, [100,10])
  assert var35.shape.as_list() == [100,10]
  var36=var14
  assert var36.shape.as_list() == [10]
  var37=tf.broadcast_to(tf.reshape(var36, [1,10]), [100,10])
  assert var37.shape.as_list() == [100,10]
  var38=tf.add(var35, var37)
  assert var38.shape.as_list() == [100,10]
  var39=tf.keras.layers.BatchNormalization(axis=1)(var38)
  assert var39.shape.as_list() == [100,10]
  var40=tf.nn.relu(var39)
  assert var40.shape.as_list() == [100,10]
  var41=tf.reshape(var40, [100,10])
  assert var41.shape.as_list() == [100,10]
  var42=var18
  assert var42.shape.as_list() == [10,2]
  var43=tf.matmul(var41, var42)
  assert var43.shape.as_list() == [100,2]
  var44=tf.reshape(var43, [100,2])
  assert var44.shape.as_list() == [100,2]
  var45=var20
  assert var45.shape.as_list() == [2]
  var46=tf.broadcast_to(tf.reshape(var45, [1,2]), [100,2])
  assert var46.shape.as_list() == [100,2]
  var47=tf.add(var44, var46)
  assert var47.shape.as_list() == [100,2]
  var48=tf.nn.softmax_cross_entropy_with_logits(labels=var21, logits=var47)
  assert var48.shape.as_list() == [100]
  var49=tf.reshape(var48, [100])
  assert var49.shape.as_list() == [100]
  var50=tf.reduce_mean(var49, axis=0)
  assert var50.shape.as_list() == []
  var51=tf.constant(0.0, shape=[], dtype=tf.float32)
  assert var51.shape.as_list() == []
  var52=tf.broadcast_to(tf.reshape(var51, [1]), [1])
  assert var52.shape.as_list() == [1]
  var53=tf.reshape(var52, [])
  assert var53.shape.as_list() == []
  var54=tf.add(var50, var53)
  assert var54.shape.as_list() == []
  var55=optimizer.minimize(var54)
  var56=tf.argmax(var47, axis=1, output_type=tf.int32)
  assert var56.shape.as_list() == [100]
  var57=tf.argmax(var21, axis=1, output_type=tf.int32)
  assert var57.shape.as_list() == [100]
  var58=tf.equal(var56, var57)
  assert var58.shape.as_list() == [100]
  var59=tf.cast(var58, tf.float32)
  assert var59.shape.as_list() == [100]
  var60=tf.reshape(var59, [100,1])
  assert var60.shape.as_list() == [100,1]
  var61=tf.constant(1.0, shape=[], dtype=tf.float32)
  assert var61.shape.as_list() == []
  var62=tf.broadcast_to(tf.reshape(var61, [1]), [1])
  assert var62.shape.as_list() == [1]
  var63=tf.reshape(var62, [1])
  assert var63.shape.as_list() == [1]
  var64=tf.broadcast_to(tf.reshape(var63, [1,1]), [100,1])
  assert var64.shape.as_list() == [100,1]
  var65=tf.concat([var60,var64], axis=1)
  assert var65.shape.as_list() == [100,2]
  var66=tf.reduce_sum(var65, axis=0)
  assert var66.shape.as_list() == [2]
  var67=tf.reshape(var47, [100,2])
  assert var67.shape.as_list() == [100,2]
  var68=tf.nn.softmax(var67, axis=1)
  assert var68.shape.as_list() == [100,2]
  var69=tf.reshape(var68, [100,2])
  assert var69.shape.as_list() == [100,2]
  var70=var0
  assert var70.shape.as_list() == []
  return {"batch_size":100
         ,"metrics":var66
         ,"y_":var69
         ,"w3_bias":var45
         ,"w3_w":var42
         ,"w2_bias":var36
         ,"w2_w":var33
         ,"w1_bias":var27
         ,"w1_w":var24
         ,"y":var21
         ,"x":var22
         ,"training_phase":var70
         ,"optimizer":optimizer
         ,"params":tf.trainable_variables()
         ,"train":var55
         ,"loss":var54
         ,"update":[]}