package com.pranav.retinascan

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.Image
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

private val Ink = Color.White
private val Panel = Paper
private val Teal = Orange
private val Muted = Grey
private val Amber = Orange
private val gradeNames = listOf("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            VisionTheme {
                val model: AnalysisViewModel = viewModel()
                val state by model.state.collectAsStateWithLifecycle()
                RetinaApp(
                    state, model::selectImage, model::analyze, model::clear,
                    model::prepareCapture, model::finishCapture, model::captureLaunchFailed
                )
            }
        }
    }
}

@Composable
internal fun ScanScreen(
    state: ScanState,
    select: (android.net.Uri) -> Unit,
    analyze: () -> Unit,
    clear: () -> Unit,
    prepareCapture: () -> android.net.Uri?,
    finishCapture: (Boolean) -> Unit,
    captureLaunchFailed: () -> Unit
) {
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) {
        if (it != null) select(it)
    }
    val camera = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) {
        finishCapture(it)
    }
    val busy = state.loadingImage || state.analyzing
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
            Column(
                Modifier.widthIn(max = 600.dp).fillMaxWidth().verticalScroll(rememberScrollState())
                    .padding(horizontal = 24.dp, vertical = 20.dp),
                verticalArrangement = Arrangement.spacedBy(24.dp)
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("EDGE INTELLIGENCE / 01", color = Teal, fontSize = 11.sp, letterSpacing = 2.sp)
                    Text("A clearer look at\nretinal health.", fontSize = 34.sp, lineHeight = 40.sp, fontWeight = FontWeight.SemiBold)
                    Text("Explore diabetic retinopathy grading from a retinal fundus image.", color = Muted, lineHeight = 23.sp)
                }
                Surface(color = Paper, shape = RoundedCornerShape(14.dp)) {
                    Row(Modifier.padding(16.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Icon(Icons.Outlined.Science, null, tint = Amber)
                        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text("Offline model ready", color = Amber, fontWeight = FontWeight.SemiBold)
                            Text("Analysis runs privately on your device using the trained retinal grading model.", color = Amber, fontSize = 12.sp, lineHeight = 18.sp)
                        }
                    }
                }
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Text("01  /  RETINAL IMAGE", color = Muted, fontSize = 11.sp, letterSpacing = 1.sp)
                        Spacer(Modifier.weight(1f))
                        if (state.image != null) TextButton(onClick = clear, enabled = !busy) { Text("Clear") }
                    }
                    Box(
                        Modifier.fillMaxWidth().height(260.dp).clip(RoundedCornerShape(22.dp))
                            .background(Panel).border(1.dp, Line, RoundedCornerShape(22.dp)),
                        contentAlignment = Alignment.Center
                    ) {
                        when {
                            state.loadingImage -> CircularProgressIndicator(color = Teal)
                            state.image != null -> Image(state.image.asImageBitmap(), "Selected retinal image", Modifier.fillMaxSize().padding(12.dp), contentScale = ContentScale.Fit)
                            else -> Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(14.dp)) {
                                Icon(Icons.Outlined.AddPhotoAlternate, null, Modifier.size(44.dp), tint = Teal)
                                Text("Start with a fundus image", fontWeight = FontWeight.Medium)
                                Text("Take a photo or upload an image", fontSize = 12.sp, color = Muted)
                            }
                        }
                    }
                    OutlinedButton(
                        onClick = {
                            prepareCapture()?.let { uri ->
                                try {
                                    camera.launch(uri)
                                } catch (e: Exception) {
                                    captureLaunchFailed()
                                }
                            }
                        },
                        enabled = !busy, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Icon(Icons.Outlined.PhotoCamera, null, Modifier.size(19.dp))
                        Spacer(Modifier.width(10.dp))
                        Text("Take photo")
                    }
                    OutlinedButton(
                        onClick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                        enabled = !busy, modifier = Modifier.fillMaxWidth().height(52.dp), shape = RoundedCornerShape(14.dp)
                    ) {
                        Icon(Icons.Outlined.PhotoLibrary, null, Modifier.size(19.dp))
                        Spacer(Modifier.width(10.dp))
                        Text("Upload image")
                    }
                    Text("Use a retinal fundus photograph. Capturing one requires suitable retinal imaging equipment; a regular eye photo is not suitable.", color = Muted, fontSize = 12.sp, lineHeight = 18.sp)
                }
                state.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                Button(
                    onClick = analyze, enabled = state.image != null && !busy,
                    modifier = Modifier.fillMaxWidth().height(56.dp), shape = RoundedCornerShape(14.dp)
                ) {
                    if (state.analyzing) {
                        CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
                        Spacer(Modifier.width(12.dp))
                        Text("Analyzing retinal image…")
                    } else {
                        Text("Analyze image", fontWeight = FontWeight.Bold)
                        Spacer(Modifier.width(12.dp))
                        Icon(Icons.Outlined.ArrowForward, null, Modifier.size(20.dp))
                    }
                }
                state.result?.let { ResultCard(it) }
                HorizontalDivider(color = Line)
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Icon(Icons.Outlined.PhonelinkLock, null, tint = Teal, modifier = Modifier.size(20.dp))
                    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                        Text("On your device. In your control.", fontSize = 13.sp, fontWeight = FontWeight.Medium)
                        Text("Images stay on your device. Camera captures use temporary cache files that are removed after loading. No cloud processing.", color = Muted, fontSize = 12.sp, lineHeight = 18.sp)
                    }
                }

            }
        }
}
@Composable
private fun ResultCard(result: AnalysisResult) {
    Surface(color = Panel, shape = RoundedCornerShape(22.dp)) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("02  /  ANALYSIS RESULT", color = Teal, fontSize = 11.sp, letterSpacing = 1.sp)
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Text(result.grade.toString(), color = Amber, fontSize = 56.sp, fontWeight = FontWeight.Light)
                Column {
                    Text(gradeNames[result.grade], fontSize = 23.sp, fontWeight = FontWeight.SemiBold)
                    Text("Grade ${result.grade} of 4", color = Muted, fontSize = 13.sp)
                }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                gradeNames.forEachIndexed { index, _ ->
                    Box(Modifier.weight(1f).height(7.dp).clip(RoundedCornerShape(4.dp))
                        .background(if (index == result.grade) Amber else Line))
                }
            }
            Text("0 · No DR     1 · Mild     2 · Moderate\n3 · Severe     4 · Proliferative DR", color = Muted, fontSize = 11.sp, lineHeight = 20.sp)
            Text("Inference: %.0f ms · On-device CPU".format(result.inferenceMs), color = Grey, fontSize = 12.sp)
            Text("Research output, not a diagnosis. Seek professional assessment for medical decisions.", color = Grey, fontSize = 12.sp, lineHeight = 19.sp)
        }
    }
}
