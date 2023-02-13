import scalanative.unsafe._
import org.ekrich.blas.unsafe.blas._
import org.ekrich.blas.unsafe.blasEnums
import scala.scalanative.runtime.struct

object Main {
  def main(args: Array[String]): Unit = {
    println("Starting blas test...")

    val N = 3
    val alpha = 2.0f
    val incX = 1
    val incY = 1
    val xt = (1, 3, -5)
    val yt = (4, -2, -1)

    {
      // using data on the stack - use caution so they remain scope
      val X = stackalloc[CFloat](3)
      X(0) = xt._1
      X(1) = xt._2
      X(2) = xt._3

      val Y = stackalloc[CFloat](3)
      Y(0) = yt._1
      Y(1) = yt._2
      Y(2) = yt._3

      // avoid string substition for g8 template
      println("X = " + xt + " Y = " + yt)
      println("CFloat routines (single precision)")

      val res2 = cblas_sdot(N, X, incX, Y, incY)
      println("cblas_sdot: " + res2)

      val res = cblas_sdsdot(N, alpha, X, incX, Y, incY)
      println("cblas_sdsdot: " + res + " alpha: " + alpha)
    }

    // using a Zone and data on the heap
    Zone { implicit z =>
      val X = alloc[CDouble](3)
      X(0) = 1
      X(1) = 3
      X(2) = -5

      val Y = alloc[CDouble](3)
      Y(0) = 4
      Y(1) = -2
      Y(2) = -1

      println("CDouble routines (double precision)")

      val res3 = cblas_ddot(N, X, incX, Y, incY)
      println("cblas_ddot: " + res3)

      val matA = Matrix.fromArray (3, 5, Array (
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 1.0f,
        3.0f, 4.0f, 5.0f, 1.0f, 2.0f
      ))
        

      val matB = Matrix.fromArray (5, 2, Array (
        -1.0f, 2.0f,
        2.0f, -3.0f,
        -3.0f, 4.0f,
        4.0f, -1.0f,
        -5.0f, 2.0f
      ))

      println (s"A = $matA")
      println (s"B = $matB")

      val matC = matA.multiply (matB)

      println (s"A x B = $matC")
    }
    println("Done.")
  }
}

class Matrix private (rows: Int, cols: Int) (implicit zone: Zone) {    
  private def _rows = rows
  private def _cols = cols

  val buffer = alloc [CFloat] (rows * cols)

  def update (i: Int, j: Int, value: Float) = buffer (i * cols + j) = value

  override def toString () = {
    val sb = new StringBuffer (100)

    sb.append (s" $rows x $cols [\n")
    for (i <- 0 until rows) {
      sb.append ("  ")
      for (j <- 0 until cols) {
        sb.append (buffer (i * cols + j).toString)
        sb.append (", ")
      }
      sb.append ("\n")
    }
    sb.append ("]\n")

    sb.toString
  }

  def multiply (other: Matrix): Matrix = {
    if (this.cols != other._rows)
      throw new RuntimeException ("Incompatibles matrix")

    val M = this.rows
    val N = other._cols
    val K = this.cols
    val lda = this.rows
    val ldb = this.cols // == other._rows
    val ldc = this.rows

    val order = blasEnums.CblasRowMajor
    val no_transpose = blasEnums.CblasNoTrans 

    val matC = new Matrix (M, N)

    cblas_sgemm(order, no_transpose, no_transpose, M, N, K, 1.0f, this.buffer, lda, other.buffer, ldb, 0.0f, matC.buffer, ldc)

    matC
  }
}

object Matrix {
  def fromArray (rows: Int, cols: Int, arr: Array [Float]) (implicit zone: Zone): Matrix = {
    if (arr.length != rows * cols) {
      throw new RuntimeException (s"Array should be of size ${rows * cols}")
    }

    val mat = new Matrix (rows, cols)

    for (i <- 0 until rows; j <- 0 until cols) {
      mat.update (i, j, arr (i * cols + j))
    }

    mat
  }
}


 
