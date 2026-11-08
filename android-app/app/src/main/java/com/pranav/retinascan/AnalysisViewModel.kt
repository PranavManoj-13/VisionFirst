package com.pranav.retinascan

import android.app.Application
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.core.content.FileProvider
import androidx.lifecycle.SavedStateHandle
import java.io.File
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import android.graphics.ColorSpace
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// The future edge runtime implements this boundary; UI never knows its tensor format.
interface RetinalAnalyzer {
    suspend fun analyze(image: Bitmap): AnalysisResult
}

data class AnalysisResult(
    val grade: Int,
    val cumulative: List<Float>,
    val classProbabilities: List<Float>,
    val inferenceMs: Double
)

data class ScanState(
    val image: Bitmap? = null,
    val loadingImage: Boolean = false,
    val analyzing: Boolean = false,
    val result: AnalysisResult? = null,
    val error: String? = null
)

class AnalysisViewModel(application: Application, private val savedState: SavedStateHandle) : AndroidViewModel(application) {
    private val analyzer: RetinalAnalyzer = LiteRtAnalyzer(application)
    private val mutableState = MutableStateFlow(ScanState())
    val state = mutableState.asStateFlow()
    private var work: Job? = null

    fun selectImage(uri: Uri) = decodeImage(uri)

    fun prepareCapture(): Uri? {
        return try {
            val app = getApplication<Application>()
            val directory = File(app.cacheDir, "captures").apply { mkdirs() }
            val file = File.createTempFile("retina_", ".jpg", directory)
            savedState["pendingCapture"] = file.name
            FileProvider.getUriForFile(app, "${app.packageName}.captureprovider", file)
        } catch (e: Exception) {
            captureLaunchFailed()
            null
        }
    }

    fun finishCapture(success: Boolean) {
        val name = savedState.remove<String>("pendingCapture") ?: return
        val app = getApplication<Application>()
        val file = File(File(app.cacheDir, "captures"), name)
        if (success) {
            val uri = FileProvider.getUriForFile(app, "${app.packageName}.captureprovider", file)
            decodeImage(uri, file)
        } else {
            file.delete() // Cancellation preserves the previous image/result.
        }
    }

    fun captureLaunchFailed() {
        finishCapture(false)
        mutableState.value = mutableState.value.copy(error = "Could not open a camera app. You can upload an image instead.")
    }

    private fun decodeImage(uri: Uri, temporaryCapture: File? = null) {
        work?.cancel()
        mutableState.value = ScanState(loadingImage = true)
        work = viewModelScope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) {
                    try {
                        val source = ImageDecoder.createSource(getApplication<Application>().contentResolver, uri)
                        ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                            require(info.size.width.toLong() * info.size.height <= 24_000_000) { "Image exceeds 24 megapixels" }
                            decoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB))
                        }
                    } finally {
                        temporaryCapture?.delete()
                    }
                }
                mutableState.value = ScanState(image = bitmap)
            } catch (e: CancellationException) { throw e
            } catch (e: Exception) {
                mutableState.value = ScanState(error = "Could not open this image. Please choose another JPG or PNG.")
            }
        }
    }

    fun analyze() {
        val current = mutableState.value
        val image = current.image ?: return
        if (current.analyzing || current.loadingImage) return
        mutableState.value = current.copy(analyzing = true, result = null, error = null)
        work = viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.Default) { analyzer.analyze(image) }
                mutableState.value = mutableState.value.copy(analyzing = false, result = result)
            } catch (e: CancellationException) { throw e
            } catch (e: Exception) {
                Log.e("VisionFirst", "LiteRT analysis failed", e)
                mutableState.value = mutableState.value.copy(analyzing = false, error = "On-device analysis failed. Please try another image or restart the app.")
            }
        }
    }

    fun clear() {
        work?.cancel()
        mutableState.value = ScanState()
    }
}
