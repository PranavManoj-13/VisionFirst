package com.pranav.retinascan

import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.Menu
import androidx.compose.material.icons.outlined.PhotoCamera
import androidx.compose.material.icons.outlined.Schema
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch

private val Background = Color.White
private val Card = Paper
private val Accent = Orange
private val Secondary = Grey

private enum class Page(val title: String, val icon: ImageVector) {
    Scan("Retinal scan", Icons.Outlined.PhotoCamera),
    About("About", Icons.Outlined.Info),
    Architecture("Architecture", Icons.Outlined.Schema)
}

@Composable
internal fun RetinaApp(
    state: ScanState,
    select: (Uri) -> Unit,
    analyze: () -> Unit,
    clear: () -> Unit,
    prepareCapture: () -> Uri?,
    finishCapture: (Boolean) -> Unit,
    captureLaunchFailed: () -> Unit
) {
    var pageName by rememberSaveable { mutableStateOf(Page.Scan.name) }
    val page = Page.valueOf(pageName)
    val drawer = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    BackHandler(enabled = page != Page.Scan && !drawer.isOpen) { pageName = Page.Scan.name }
    ModalNavigationDrawer(
        drawerState = drawer,
        drawerContent = {
            ModalDrawerSheet(drawerContainerColor = Background, modifier = Modifier.widthIn(max = 320.dp)) {
                Column(
                    Modifier.fillMaxHeight().verticalScroll(rememberScrollState()).padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Image(painterResource(R.drawable.ic_retinascan), "VisionFirst logo", Modifier.size(72.dp))
                    Text("VisionFirst", fontSize = 26.sp, fontWeight = FontWeight.SemiBold)
                    Text("EDGE INTELLIGENCE", color = Accent, fontSize = 11.sp, letterSpacing = 2.sp)
                    Spacer(Modifier.height(16.dp))
                    Page.entries.forEach { destination ->
                        NavigationDrawerItem(
                            label = { Text(destination.title) },
                            icon = { Icon(destination.icon, null) },
                            selected = destination == page,
                            onClick = {
                                pageName = destination.name
                                scope.launch { drawer.close() }
                            },
                            colors = NavigationDrawerItemDefaults.colors(
                                selectedContainerColor = Card, selectedIconColor = Accent,
                                selectedTextColor = Accent, unselectedContainerColor = Background
                            )
                        )
                    }
                    Spacer(Modifier.height(24.dp))
                    HorizontalDivider(color = Card)
                    Text("Research prototype · Offline inference", color = Secondary, fontSize = 12.sp)
                    Credit()
                }
            }
        }
    ) {
        Scaffold(
            containerColor = Background,
            topBar = {
                Surface(color = Background, shadowElevation = 0.dp, border=androidx.compose.foundation.BorderStroke(1.dp, Line)) {
                    Row(
                        Modifier.fillMaxWidth().statusBarsPadding().padding(horizontal = 8.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        IconButton(onClick = { scope.launch { drawer.open() } }) {
                            Icon(Icons.Outlined.Menu, "Open navigation menu", tint = Accent)
                        }
                        Image(painterResource(R.drawable.ic_retinascan), null, Modifier.size(44.dp))
                        Column(Modifier.weight(1f).padding(start = 6.dp)) {
                            Text("VisionFirst", fontSize = 20.sp, fontWeight = FontWeight.SemiBold)
                            Text(page.title, color = Secondary, fontSize = 12.sp)
                        }
                        Surface(color = Card, shape = RoundedCornerShape(8.dp)) {
                            Text("ON DEVICE", Modifier.padding(horizontal = 10.dp, vertical = 6.dp), color = Accent, fontSize = 10.sp)
                        }
                        Spacer(Modifier.width(12.dp))
                    }
                }
            }
        ) { padding ->
            Box(Modifier.fillMaxSize().padding(padding)) {
                when (page) {
                    Page.Scan -> ScanScreen(state, select, analyze, clear, prepareCapture, finishCapture, captureLaunchFailed)
                    Page.About -> AboutPage()
                    Page.Architecture -> ArchitecturePage()
                }
            }
        }
    }
}

@Composable
private fun InfoPage(eyebrow: String, title: String, intro: String, content: @Composable ColumnScope.() -> Unit) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
        Column(
            Modifier.widthIn(max = 680.dp).fillMaxWidth().verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp, vertical = 24.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            Text(eyebrow, color = Accent, fontSize = 11.sp, letterSpacing = 2.sp)
            Text(title, fontSize = 32.sp, lineHeight = 38.sp, fontWeight = FontWeight.SemiBold)
            Text(intro, color = Secondary, lineHeight = 24.sp)
            content()
            HorizontalDivider(color = Card)
        }
    }
}

@Composable
private fun Detail(title: String, body: String) {
    Surface(color = Card, shape = RoundedCornerShape(18.dp), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title, color = Accent, fontWeight = FontWeight.SemiBold, fontSize = 17.sp)
            Text(body, color = Secondary, fontSize = 14.sp, lineHeight = 23.sp)
        }
    }
}

@Composable
private fun Credit() {
    Text(stringResource(R.string.project_credit), color = Secondary, fontSize = 12.sp, lineHeight = 19.sp)
}

@Composable
private fun AboutPage() {
    InfoPage(
        "ABOUT THE PROJECT", "Retinal insight.\nAt the edge.",
        "VisionFirst explores how a lightweight neural network can grade diabetic retinopathy from retinal fundus photographs directly on an Android device."
    ) {
        Detail("What the app does", "Take a photo with suitable retinal imaging equipment or upload an existing fundus image, preview it, and follow the image-to-grade workflow. An ordinary photograph of the outside of an eye is not a fundus image.")
        Detail("Five ordered severity grades", "0 · No DR\n1 · Mild\n2 · Moderate\n3 · Severe\n4 · Proliferative DR\n\nThe model is designed to respect the ordering of these grades, rather than treating them as unrelated categories.")
        Detail("Why edge intelligence?", "The goal is local inference without a server round trip. A compact model can reduce dependence on connectivity and keep images on the device. Actual speed and memory use must be measured on the target phone.")
        Detail("Real on-device inference", "This build bundles the trained MobileNetV3 ordered model. Images are analyzed locally using LiteRT on the CPU, without a server or a demo fallback. The original held-out test run measured 74.4% exact-grade accuracy and 0.884 QWK; these research results do not establish clinical suitability.")
        Detail("Privacy and local processing", "No accounts, image uploads, analytics, or network permission are used. Selected images are held in memory. Camera images use temporary app-cache files, removed after loading or cancellation. The session resets if Android terminates the process.")
        Detail("Research scope", "Built for DSC4109: Edge Intelligence and Automation. This is an educational research prototype, not a diagnostic tool or a replacement for professional eye assessment.")
    }
}

@Composable
private fun ArchitecturePage() {
    InfoPage(
        "MODEL ARCHITECTURE", "From pixels\nto ordered grades.",
        "The current training pipeline uses torchvision MobileNetV3-Large with an ordered threshold head. The bundled LiteRT model runs on the device CPU with the same ordered-threshold prediction rule used in Python."
    ) {
        Detail("01 / Input and preprocessing", "Default input: 224 × 224 RGB. Decode image orientation, convert to RGB, and bilinearly resize to a square.\n\nThe mobile contract is float32 [1, 224, 224, 3], with raw values from 0 to 255. The model converts to channel-first layout, divides by 255, and normalizes with ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. The app must not normalize twice.\n\nThe new pipeline uses no circular crop, Ben Graham enhancement, or CLAHE.")
        Detail("02 / MobileNetV3-Large backbone", "Training starts from ImageNet pretrained weights. Inverted residual blocks and depthwise convolutions extract features efficiently. Built-in squeeze-and-excitation blocks reweight feature channels; selected blocks use hard-swish activations. There is no additional attention MLP.\n\nAt 224 × 224, the final feature map is 960 × 7 × 7. Global average pooling reduces it to 960 features.")
        Detail("03 / Feature projection", "A linear layer maps 960 pooled features to 1,280 features, followed by hard-swish and dropout (0.2 during training). A shared linear projection then produces one scalar severity score.")
        Detail("04 / Ordered threshold head", "Four learned, increasing cut points divide the severity axis. The first cut point is learned directly; later gaps use softplus plus a small positive offset so they cannot become negative.\n\nEach logit is severity score minus cut point. A sigmoid gives P(grade > 0), P(grade > 1), P(grade > 2), and P(grade > 3). These cumulative probabilities remain in descending order. This is ordered threshold regression, rather than four independent class heads.")
        Detail("05 / Grade and probabilities", "The predicted grade is the number of cumulative probabilities strictly greater than 0.5. Example: [0.94, 0.76, 0.21, 0.08] produces Grade 2.\n\nIf cumulative values are p0…p3, the five class probabilities are [1−p0, p0−p1, p1−p2, p2−p3, p3]. They are nonnegative and sum to one.")
        Detail("06 / Training and calibration", "Labels become four binary targets: Grade 2 is [1, 1, 0, 0]. Training uses binary cross-entropy over these ordinal targets, AdamW, and a cosine learning-rate schedule. Defaults: 20 epochs, learning rate 0.0003, weight decay 0.0001.\n\nThe best checkpoint is selected by validation quadratic weighted kappa (QWK), with early stopping after five epochs without meaningful improvement. A positive temperature is fitted using validation BCE. Temperature adjusts probabilities but preserves the grade at the 0.5 threshold. These values are not established clinical confidence estimates.")
        Detail("07 / Data isolation and evaluation", "Approximately 60% training, 20% validation, and 20% test, stratified by grade with pixel-identical images grouped together. Conflicting-label duplicate groups are excluded and reported. The current default split contains 2,159 / 718 / 723 images after 62 conflicting images are excluded.\n\nTest is used after model selection and calibration. Reported metrics include QWK, accuracy, confusion counts, and AUC per threshold. Screening sensitivity and specificity use referable DR: Grade 2 or above. Patient IDs are unavailable, so near-duplicate and patient overlap still require a separate audit.")
        Detail("08 / Android deployment path", "Train on the Mac GPU with MPS, then convert the checkpoint to a float32 LiteRT .tflite model using the Linux converter. Export checks input/output shapes, probability agreement, and grade agreement against Python fixtures.\n\nThis build integrates float32 CPU inference. Golden-input checks compare Android outputs against Python fixtures. Inference duration is measured on each analysis; full phone memory and performance evaluation remains separate. Quantization follows float32 parity and requires new accuracy measurements. TorchScript export is available for reference checks, but is not the chosen Android artifact.")
    }
}
