package com.pranav.retinascan

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import org.junit.Assert.*
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(AndroidJUnit4::class)
class ModelParityTest {
    private fun floats(name: String): FloatArray {
        val bytes = InstrumentationRegistry.getInstrumentation().context.assets.open(name).use { it.readBytes() }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }
    @Test fun resizeMatchesPillow() {
        val context=InstrumentationRegistry.getInstrumentation().context
        val bitmap=context.assets.open("resize_source.png").use { android.graphics.BitmapFactory.decodeStream(it) }
        for (size in listOf(224,11)) {
            val expected=floats("resize_$size.bin")
            val actual=RgbPreprocessor.resize(bitmap,size)
            assertArrayEquals(expected,actual,0f)
        }
    }
    @Test fun exportedInputsMatchPython() {
        val analyzer=LiteRtAnalyzer(InstrumentationRegistry.getInstrumentation().targetContext)
        for (i in 0..4) {
            val expected=floats("output_$i.bin")
            val result=analyzer.analyzeRgb(floats("input_$i.bin"))
            for (k in 0..3) assertEquals(expected[k],result.cumulative[k],1e-4f)
            assertEquals(expected.count { it>.5f }, result.grade)
        }
    }
}
