import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import java.io.{DataOutputStream, ByteArrayOutputStream}
import org.apache.hadoop.io.WritableComparator
import scala.collection.mutable.ArrayBuffer

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

  class KeyValueChunk(size: Int = 4296) extends ByteArrayOutputStream(size) {
    def join(key:Long,value:Double): ArrayBuffer[(Long,Double)] = {
      val res = new ArrayBuffer[(Long, Double)]
      var i = 0
      var offset:Int = 0
      val length = WritableComparator.readLong(buf,offset)
      var currentKey:Long=0
      while ((i<length)&&(offset <= count)) {
        offset += 8
        currentKey = WritableComparator.readLong(buf, offset)
        offset += 8
        val valueLength=WritableComparator.readLong(buf, offset)

        if (currentKey == key) {
          var j= 0
          var currentValue:Long=0
          while ((j<valueLength)&&(offset <= count)){
            offset += 8
            currentValue = WritableComparator.readLong(buf, offset)
            res += ((currentValue,value/valueLength))
            j += 1
          }
          return res
        }
        offset += 8*valueLength.toInt
        i+=1
      }
      null
    }
 }

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

    var iters = 1
    val ctx = new SparkContext(sparkConf)
    val lines = ctx.textFile("hdfs://11.11.0.55:9000//HiBench/Pagerank/Input/edges")
    val links = lines.map{ s =>
      val parts = s.split("\\s+")
      (parts(0).toLong, parts(1).toLong)
    }.groupByKey()

    var ranks = links.mapValues(v => 1.0)

     val cachedData = links.mapPartitions { iter =>
        val chunk = new KeyValueChunk(42960)
        val dos = new DataOutputStream(chunk)
        val (iter1, iter2) = iter.duplicate
        dos.writeLong(iter1.size)
        iter2.foreach(x => {
          dos.writeLong(x._1)
          dos.writeLong(x._2.size)
          x._2.foreach(y => dos.writeLong(y))
        })
        Iterator(chunk)
      }.persist(StorageLevel.MEMORY_ONLY)

       def zipPartitionsFunc(l: Iterator[KeyValueChunk], r: Iterator[(Long, Double)]): Iterator[(Long, Double)] = {
        val res = new ArrayBuffer[(Long, Double)]
        if (l.hasNext) {
          val chunkData = l.next()
          while (r.hasNext) {
            val keyValue = r.next()
            res ++= chunkData.join(keyValue._1, keyValue._2)
          }
          res.iterator
                                                                                                                                      46,1          75%
 }
        else
          null
      }

    for (i <- 1 to iters) {
      val contribs = cachedData.zipPartitions(ranks)(zipPartitionsFunc)
      ranks = contribs.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _)
    }

   ranks.collect()
   // ranks.saveAsTextFile("hdfs://11.11.0.55:9000//user/root/output")
//    ctx.stop()
  }
}
                
