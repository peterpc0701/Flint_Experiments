import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel

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
      (parts(0), parts(1))
    }.groupByKey().persist(StorageLevel.MEMORY_AND_DISK_SER)
    //links.collect()
    var ranks = links.mapValues(v => 1.0)
    for (i <- 1 to iters) {
      val contribs = links.join(ranks).values.flatMap{ case (urls, rank) =>
        val size = urls.size
        urls.map(url => (url, rank / size))
      }
      ranks = contribs.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _)
    }

   ranks.collect()
   // ranks.saveAsTextFile("hdfs://11.11.0.55:9000//user/root/output")
//    ctx.stop()
  }
}

