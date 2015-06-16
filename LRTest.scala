package cn.edu.hust

/**
 * Created by peicheng on 15-6-11.
 */

import org.apache.spark._

import java.util.Random

import org.apache.spark.rdd.RDD

import scala.math.exp

import breeze.linalg.{Vector, DenseVector}

import java.io.{DataOutputStream, ByteArrayOutputStream}

import org.apache.hadoop.io.WritableComparator

class PointChunk(dimensions: Int,size: Int = 4196) extends ByteArrayOutputStream(size) { self =>

  def getVectorValueIterator(w: Array[Double]) = new Iterator[Array[Double]] {
    var offset = 0
    var currentPoint=new Array[Double](dimensions)
    var i = 0
    var y = 0.0
    var dotvalue = 0.0

    override def hasNext = offset < self.count

    override def next() = {
      if (!hasNext) Iterator.empty.next()
      else {
        //read data from the chunk
        i=0
        while (i < dimensions) {
          currentPoint(i)= WritableComparator.readDouble(buf, offset)
          offset += 8
          i += 1
        }
        y = WritableComparator.readDouble(buf, offset)
        offset += 8
        //calculate the dot value
        i=0
        dotvalue = 0.0
        while (i < dimensions) {
          dotvalue += w(i)*currentPoint(i)
          i += 1
        }
        //transform to values
        i=0
        while (i < dimensions) {
          currentPoint(i) *= (1 / (1 + exp(-y * dotvalue)) - 1) * y
          i += 1
        }
        currentPoint.clone()
      }
    }
  }
}
/**
 * Logistic regression based classification.
 * Usage: SparkLR [slices]
 *
 * This is an example implementation for learning how to use Spark. For more conventional use,
 * please refer to either org.apache.spark.mllib.classification.LogisticRegressionWithSGD or
 * org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS based on your needs.
 */

object LRTest {
  val N = 1000  // Number of data points
  val D = 100   // Numer of dimensions
  val R = 0.7  // Scaling factor
  val ITERATIONS = 5
  val rand = new Random(42)

  case class DataPoint(x: Vector[Double], y: Double)

  // Initialize w to a random value
  var w = DenseVector.fill(D){2 * rand.nextDouble - 1}
  println("Initial w: " + w)

  def generateData(size: Int) = {
    def generatePoint(i: Int) = {
      val y = if(i % 2 == 0) -1 else 1
      val x = DenseVector.fill(D){rand.nextGaussian + y * R}
      DataPoint(x, y)
    }
    Array.tabulate(size)(generatePoint)
  }

  def testNative(points: RDD[DataPoint]): Unit = {
    points.cache()
    val startTime = System.currentTimeMillis
    for (i <- 1 to ITERATIONS) {
      println("On iteration " + i)
      val gradient = points.map { p =>
        p.x * (1 / (1 + exp(-p.y * (w.dot(p.x)))) - 1) * p.y
      }.reduce(_ + _)
      w -= gradient
    }
    val duration = System.currentTimeMillis - startTime
    println("Duration is " + duration / 1000.0 + " seconds")
    // println("Final w: " + w.length)
    //println("Final w: " + w)
  }

  def testOptimized(points: RDD[DataPoint]): Unit = {
    val cachedPoints = points.mapPartitions { iter =>
      val (iterOne ,iterTwo) = iter.duplicate
      val chunk = new PointChunk(D,8*iterOne.length*(1+D))
      val dos = new DataOutputStream(chunk)
      for (point <- iterTwo) {
        point.x.foreach(dos.writeDouble)
        dos.writeDouble(point.y)
      }
      Iterator(chunk)
    }.cache()

    val w_op=new Array[Double](D)
    for(i <- 0 to D-1)
      w_op(i) = w(i)

    val startTime = System.currentTimeMillis
    for (i <- 1 to ITERATIONS) {
      println("On iteration " + i)
      val gradient= cachedPoints.mapPartitions{ iter =>
        val chunk = iter.next()
        chunk.getVectorValueIterator(w_op)
      }.reduce{(lArray, rArray) =>
        val result_array=new Array[Double](lArray.length)
        for(i <- 0 to D-1)
          result_array(i) = lArray(i) + rArray(i)
        result_array
      }

      for(i <- 0 to D-1)
        w_op(i) = w_op(i) - gradient(i)
    }
    val duration = System.currentTimeMillis - startTime
    println("Duration is " + duration / 1000.0 + " seconds")

  }

  def main(args: Array[String]) {

    val sparkConf = new SparkConf().setAppName("SparkLR").setMaster("local")
    val sc = new SparkContext(sparkConf)
    val numSlices = if (args.length > 0) args(0).toInt else 2
    val points = sc.parallelize(generateData(N), numSlices)

    //test the original version
    //testNative(points)

    //test the manual version
    testOptimized(points)

    sc.stop()
  }
}
