import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import java.io.{DataOutputStream, ByteArrayOutputStream}
import org.apache.hadoop.io.WritableComparator
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.rdd.{RDD, ShuffledRDD}

/**
 * Computes the PageRank of URLs from an input file. Input file should
 * be in format of:
 * URL         neighbor URL
 * URL         neighbor URL
 * URL         neighbor URL
 * ...
 * where URL and their neighbors are separated by space(s).
 */
object PageRank {

  class EdgeChunk(size: Int = 4196) extends ByteArrayOutputStream(size) { self =>

  def show() {
    var offset = 0
    while (offset < count) {
      println("src id: " + WritableComparator.readLong(buf, offset))
      offset += 8
      val numDests = WritableComparator.readInt(buf, offset)
      offset += 4
      print("has " + numDests + " dests:")
      var count = 0
      while (count < numDests) {
        print(" " + WritableComparator.readLong(buf, offset))
        offset += 8
        count += 1
      }
      println("")
    }
  }

  def getInitValueIterator(value: Double) = new Iterator[(Long, Double)] {
    var offset = 0

    override def hasNext = offset < self.count

    override def next() = {
      if (!hasNext) Iterator.empty.next()
      else {
        val srcId = WritableComparator.readLong(buf, offset)
        offset += 8
        val numDests = WritableComparator.readInt(buf, offset)
        offset += 4 + 8 * numDests
        (srcId, value)
      }
    }
  }

  def getMessageIterator(vertices: Iterator[(Long, Double)]) = new Iterator[(Long, Double)] {
    var changeVertex = true
    var currentVertex: (Long, Double) = _
    var offset = 0
    var currentDestIndex = 0
    var currentDestNum = 0

    override def hasNext = !changeVertex || vertices.hasNext

    override def next() = {
      if (!hasNext) Iterator.empty.next()
      else {
        if (changeVertex) {
          currentVertex = vertices.next()
          while (currentVertex._1 != WritableComparator.readLong(buf, offset)) {
            offset += 8
            val numDests = WritableComparator.readInt(buf, offset)
            offset += 4 + 8 * numDests
          }
          offset += 8
          currentDestNum = WritableComparator.readInt(buf, offset)
          offset += 4
          currentDestIndex = 0
          changeVertex = false
        }

        currentDestIndex += 1
        if (currentDestIndex == currentDestNum) changeVertex = true

        val destId = WritableComparator.readLong(buf, offset)
        offset += 8

        (destId, currentVertex._2)
      }
    }

  }
}
  private val ordering = implicitly[Ordering[Long]]
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("PageRank")
                                  .set("spark.executor.memory", "40g")
                                  .set("spark.shuffle.consolidateFiles", "true")
                                 .set("spark.shuffle.use.netty","true")
                                .set("spark.shuffle.memoryFraction", "0.2")
                                 .set("spark.shuffle.file.buffer.kb", "32")
                                 .set("spark.reducer.maxMbInFlight","48")
                                 .set("spark.io.compression.codec", "snappy")
                                // .set("spark.default.parallelism", "512")

    var iters = 3
    val ctx = new SparkContext(sparkConf)
    val lines = ctx.textFile("hdfs://11.11.0.55:9000//HiBench/Pagerank/Input/edges")
    val links = lines.map{ s =>
      val parts = s.split("\\s+")
      (parts(0).toLong, parts(1).toLong)
    }.groupByKey().
      asInstanceOf[ShuffledRDD[Long, _, _]].
      setKeyOrdering(ordering).
      asInstanceOf[RDD[(Long, Iterable[Long])]]

   val cachedEdges = links.mapPartitions { iter =>
      val chunk = new EdgeChunk
      val dos = new DataOutputStream(chunk)
      for ((src, dests) <- iter) {
        dos.writeLong(src)
        dos.writeInt(dests.size)
        dests.foreach(dos.writeLong)
      }
      Iterator(chunk)
    }.cache()

    cachedEdges.foreach(_ => Unit)

    val initRanks = cachedEdges.mapPartitions{ iter =>
      val chunk = iter.next()
      chunk.getInitValueIterator(1.0)
    }

    var ranks = initRanks

    val startTime = System.currentTimeMillis
    for (i <- 1 to iters) {
      val contribs = cachedEdges.zipPartitions(ranks) { (EIter, VIter) =>
        val chunk = EIter.next()
        chunk.getMessageIterator(VIter)
      }
      ranks = contribs.reduceByKey(_ + _).asInstanceOf[ShuffledRDD[Long, _, _]].
        setKeyOrdering(ordering).
        asInstanceOf[RDD[(Long, Double)]].
        mapValues(0.15 + 0.85 * _)
    }
    //ranks.foreach(_ => Unit)
    ranks.collect()
    val duration = System.currentTimeMillis - startTime
    println("Duration is " + duration / 1000.0 + " seconds")
  }
}
                
