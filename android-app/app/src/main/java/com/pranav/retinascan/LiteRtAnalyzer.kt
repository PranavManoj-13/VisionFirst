package com.pranav.retinascan

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.max

/** Offline, float32 CPU inference. No network or demo fallback. */
class LiteRtAnalyzer(private val context: Context) : RetinalAnalyzer {
    override suspend fun analyze(image: Bitmap): AnalysisResult = analyzeRgb(RgbPreprocessor.resize(image, 224))

    fun analyzeRgb(rgb: FloatArray): AnalysisResult {
        require(rgb.size == 224 * 224 * 3 && rgb.all { it.isFinite() && it in 0f..255f })
        val metadata = JSONObject(context.assets.open("model_metadata.json").bufferedReader().use { it.readText() })
        check(metadata.getString("architecture") == "mobilenet_v3_large_ordered_v2")
        val bytes = context.assets.open("visionfirst.tflite").use { it.readBytes() }
        val hash = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        check(hash == metadata.getString("sha256")) { "The bundled model failed its integrity check." }
        val modelBuffer=ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply { put(bytes); rewind() }
        Interpreter(modelBuffer, Interpreter.Options().setNumThreads(2)).use { model ->
            check(model.getInputTensor(0).shape().contentEquals(intArrayOf(1,224,224,3)))
            check(model.getOutputTensor(0).shape().contentEquals(intArrayOf(1,4)))
            check(model.getInputTensor(0).dataType()==DataType.FLOAT32)
            check(model.getOutputTensor(0).dataType()==DataType.FLOAT32)
            val input=ByteBuffer.allocateDirect(rgb.size*4).order(ByteOrder.nativeOrder())
            input.asFloatBuffer().put(rgb)
            val output=Array(1) { FloatArray(4) }
            val start=SystemClock.elapsedRealtimeNanos()
            model.run(input,output)
            val elapsed=(SystemClock.elapsedRealtimeNanos()-start)/1_000_000.0
            val cumulative=output[0]
            check(cumulative.all { it.isFinite() && it in -1e-5f..1.00001f })
            check((0..2).all { cumulative[it]+1e-5>=cumulative[it+1] })
            val p=cumulative.map { it.coerceIn(0f,1f) }
            val classes=listOf(1-p[0],p[0]-p[1],p[1]-p[2],p[2]-p[3],p[3]).map { it.coerceAtLeast(0f) }
            return AnalysisResult(p.count { it>.5f },p,classes,elapsed)
        }
    }
}

/** Separable antialiased bilinear RGB resize, matching Pillow's 8-bit pipeline. */
object RgbPreprocessor {
    private data class Filter(val start: Int, val weights: IntArray)
    private fun filters(source: Int, target: Int): List<Filter> {
        val scale = source.toDouble() / target
        val support = max(1.0, scale)
        return List(target) { index ->
            val center = (index + .5) * scale
            val start = (center - support + .5).toInt().coerceAtLeast(0)
            val end = (center + support + .5).toInt().coerceAtMost(source)
            val weights = DoubleArray(end-start) { i -> max(0.0, 1-abs((start+i+.5-center)/support)) }
            val total = weights.sum()
            Filter(start, IntArray(weights.size) { (weights[it]/total * (1 shl 22) + .5).toInt() })
        }
    }
    fun resize(bitmap: Bitmap, size: Int): FloatArray {
        val width=bitmap.width; val height=bitmap.height
        val pixels=IntArray(width*height);bitmap.getPixels(pixels,0,width,0,0,width,height)
        val horizontal=filters(width,size);val vertical=filters(height,size)
        val intermediate=IntArray(size*height*3)
        for (y in 0 until height) for (x in 0 until size) for (channel in 0..2) {
            val filter=horizontal[x];var sum=1L shl 21
            for (i in filter.weights.indices) {
                val pixel=pixels[y*width+filter.start+i]
                sum += ((pixel shr (16-channel*8)) and 255).toLong()*filter.weights[i]
            }
            intermediate[(y*size+x)*3+channel]=(sum shr 22).toInt().coerceIn(0,255)
        }
        return FloatArray(size*size*3) { index ->
            val channel=index%3;val x=(index/3)%size;val y=index/(size*3)
            val filter=vertical[y];var sum=1L shl 21
            for (i in filter.weights.indices) sum+=intermediate[((filter.start+i)*size+x)*3+channel].toLong()*filter.weights[i]
            (sum shr 22).toInt().coerceIn(0,255).toFloat()
        }
    }
}
